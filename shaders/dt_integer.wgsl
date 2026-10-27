// Exact modular signs. Each workgroup evaluates one pending determinant
// modulo 96 16-bit primes, then Garner mixed-radix digits distinguish its sign.
// Product(primes)>2^1440 > 2*maximum finite-f32 insphere determinant (2^1397).
// Orientation needs 56 primes (>2^840). Products of two residues fit u32.
var<workgroup> residues:array<u32,96>;
var<workgroup> cold_active:u32;
var<workgroup> cold_count:u32;
var<workgroup> cold_shift:u32;
var<workgroup> cold_result:i32;
// A floating quotient estimate is corrected with exact u32 arithmetic. For
// these 16-bit primes, binary32 division error is much smaller than one integer
// quotient unit, so at most one add and one subtract recover the exact residue.
fn mod_reduce(value:u32,prime:u32)->u32{
    let quotient=u32(f32(value)/f32(prime));var remainder=value-quotient*prime;
    if((remainder&0x80000000u)!=0u){remainder+=prime;}if(remainder>=prime){remainder-=prime;}return remainder;
}
fn mod_mul(a:u32,b:u32,prime:u32)->u32{return mod_reduce(a*b,prime);}
fn mod_add(a:u32,b:u32,prime:u32)->u32{let sum=a+b;return select(sum,sum-prime,sum>=prime);}
fn mod_sub(a:u32,b:u32,prime:u32)->u32{return select(a+prime-b,a-b,a>=b);}
fn integer_mod(bits:u32,index:u32,shift:u32)->u32{
    let prime=prime_table[index];let exponent=(bits>>23u)&255u;let fraction=bits&0x7fffffu;
    var mantissa=select(fraction,fraction|0x800000u,exponent!=0u);let power=select(0u,exponent-1u,exponent!=0u);var value=0u;
    if(mantissa!=0u){
        if(power<shift){mantissa=mantissa>>(shift-power);value=mod_reduce(mantissa,prime);}
        else{value=mod_mul(mod_reduce(mantissa,prime),prime_table[9312u+index*256u+power-shift],prime);}
    }
    return select(value,select(prime-value,0u,value==0u),(bits>>31u)!=0u);
}
fn integer_point(id:u32,index:u32,shift:u32)->vec3<u32>{let b=coordinates[id];return vec3<u32>(integer_mod(b.x,index,shift),integer_mod(b.y,index,shift),integer_mod(b.z,index,shift));}
fn mod_delta(a:vec3<u32>,b:vec3<u32>,prime:u32)->vec3<u32>{return vec3<u32>(mod_sub(a.x,b.x,prime),mod_sub(a.y,b.y,prime),mod_sub(a.z,b.z,prime));}
fn mod_det(a:vec3<u32>,b:vec3<u32>,c:vec3<u32>,prime:u32)->u32{
    let x=mod_sub(mod_mul(b.y,c.z,prime),mod_mul(b.z,c.y,prime),prime);
    let y=mod_sub(mod_mul(b.z,c.x,prime),mod_mul(b.x,c.z,prime),prime);
    let z=mod_sub(mod_mul(b.x,c.y,prime),mod_mul(b.y,c.x,prime),prime);
    return mod_add(mod_add(mod_mul(a.x,x,prime),mod_mul(a.y,y,prime),prime),mod_mul(a.z,z,prime),prime);
}
fn mod_lift(a:vec3<u32>,prime:u32)->u32{return mod_add(mod_add(mod_mul(a.x,a.x,prime),mod_mul(a.y,a.y,prime),prime),mod_mul(a.z,a.z,prime),prime);}
fn determinant_mod(c:Cache,index:u32,shift:u32)->u32{
    let prime=prime_table[index];let origin=integer_point(select(c.ids.x,c.more.x,c.more.y==1u),index,shift);
    if(c.more.y==0u){return mod_det(mod_delta(integer_point(c.ids.y,index,shift),origin,prime),mod_delta(integer_point(c.ids.z,index,shift),origin,prime),mod_delta(integer_point(c.ids.w,index,shift),origin,prime),prime);}
    let a=mod_delta(integer_point(c.ids.x,index,shift),origin,prime);let b=mod_delta(integer_point(c.ids.y,index,shift),origin,prime);
    let d=mod_delta(integer_point(c.ids.w,index,shift),origin,prime);let e=mod_delta(integer_point(c.ids.z,index,shift),origin,prime);
    let positive=mod_add(mod_mul(mod_lift(b,prime),mod_det(a,e,d,prime),prime),mod_mul(mod_lift(d,prime),mod_det(a,b,e,prime),prime),prime);
    let negative=mod_add(mod_mul(mod_lift(a,prime),mod_det(b,e,d,prime),prime),mod_mul(mod_lift(e,prime),mod_det(a,b,d,prime),prime),prime);
    return mod_sub(positive,negative,prime);
}
// Remove the common exact power of two before choosing a CRT modulus. Typical
// projected sites need far fewer primes than the worst-case finite-f32 bound.
// M-bit coordinates imply |orient|<2^(3M+6), |insphere|<2^(5M+12).
fn exact_range(c:Cache)->vec2<u32>{
    var shift=300u;var top=0u;let n=select(4u,5u,c.more.y==1u);
    for(var i=0u;i<n;i++){
        let id=select(c.ids[i%4u],c.more.x,i==4u);let b=coordinates[id];
        for(var k=0u;k<3u;k++){
            let exponent=(b[k]>>23u)&255u;let mantissa=select(b[k]&0x7fffffu,(b[k]&0x7fffffu)|0x800000u,exponent!=0u);
            if(mantissa!=0u){let power=select(0u,exponent-1u,exponent!=0u);shift=min(shift,power+countTrailingZeros(mantissa));top=max(top,power+firstLeadingBit(mantissa)+1u);}
        }
    }
    if(top==0u){return vec2<u32>(0u,1u);}let bits=top-shift;
    let bound=select(3u*bits+6u,5u*bits+12u,c.more.y==1u);
    return vec2<u32>(shift,min(96u,(bound+16u)/15u));
}
fn crt_value(c:Cache,lane:u32)->i32{
    if(lane==0u){let range=exact_range(c);cold_shift=range.x;cold_count=range.y;}
    workgroupBarrier();
    let count=workgroupUniformLoad(&cold_count);let shift=workgroupUniformLoad(&cold_shift);
    // Uniform loop bounds avoid any divergence around the CRT barriers.
    for(var i=lane;i<count;i+=POST_LANES){residues[i]=determinant_mod(c,i,shift);}
    workgroupBarrier();
    for(var j=0u;j<count;j++){
        let digit=residues[j];
        for(var i=lane;i<count;i+=POST_LANES){if(i>j){let prime=prime_table[i];let difference=mod_sub(residues[i],mod_reduce(digit,prime),prime);residues[i]=mod_mul(difference,prime_table[96u+j*96u+i],prime);}}
        workgroupBarrier();
    }
    if(lane==0u){
        var sign=0i;var nonzero=false;
        for(var j=0u;j<count;j++){nonzero=nonzero||residues[j]!=0u;}
        if(nonzero){sign=1;for(var j=i32(count)-1;j>=0;j--){let middle=(prime_table[u32(j)]-1u)/2u;if(residues[u32(j)]!=middle){sign=select(1,-1,residues[u32(j)]>middle);break;}}}
        cold_result=sign;
    }
    workgroupBarrier();return workgroupUniformLoad(&cold_result);
}

