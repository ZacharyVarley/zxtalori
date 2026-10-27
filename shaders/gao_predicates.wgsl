// Independent adaptive predicates use a transposed shared CRT workspace in
// the post pass and a private workspace in the output pass.
// Ordinary signs use binary32, then double-single; exact integer work is cold.
var<private> gao_exact_count:u32;
var<private> gao_predicate_overflow:bool;
var<private> gao_exact_zeros:u32;
var<private> gao_prime_sum:u32;
var<private> gao_prime_max:u32;
var<private> gao_exact_orientations:u32;
var<private> gao_exact_spheres:u32;
var<private> gao_defer_exact:bool;
var<private> gao_private_digits:array<u32,GAO_INTEGER_PRIMES>;
const GAO_SHARED_DIGIT_WORDS=select(1u,POST_LANES*GAO_INTEGER_PRIMES,POST_LANES<=32u);
var<workgroup> gao_shared_digits:array<u32,GAO_SHARED_DIGIT_WORDS>;
// Adjacent lanes access adjacent words. Lane-major storage with a 32/96-word
// stride would send an entire warp to one shared-memory bank at every digit.
fn gao_digit(i:u32)->u32{if(GAO_POST&&POST_LANES<=32u){return gao_shared_digits[i*POST_LANES+gao_lane];}return gao_private_digits[i];}
fn gao_digit_put(i:u32,value:u32){if(GAO_POST&&POST_LANES<=32u){gao_shared_digits[i*POST_LANES+gao_lane]=value;}else{gao_private_digits[i]=value;}}
fn gao_integer_sign(c:Cache)->i32{
    gao_exact_count++;
    let range=exact_range(c);let count=range.y;
    if((p.reserved&2u)!=0u){gao_prime_sum+=count;gao_prime_max=max(gao_prime_max,count);if(c.more.y==0u){gao_exact_orientations++;}else{gao_exact_spheres++;}}
    if(count>GAO_INTEGER_PRIMES){gao_predicate_overflow=true;return 2;}
    for(var i=0u;i<count;i++){gao_digit_put(i,determinant_mod(c,i,range.x));}
    for(var j=0u;j<count;j++){
        let digit=gao_digit(j);
        for(var i=j+1u;i<count;i++){let prime=prime_table[i];gao_digit_put(i,mod_mul(mod_sub(gao_digit(i),mod_reduce(digit,prime),prime),prime_table[96u+j*96u+i],prime));}
    }
    var nonzero=false;for(var j=0u;j<count;j++){nonzero=nonzero||gao_digit(j)!=0u;}
    if(!nonzero){if((p.reserved&2u)!=0u){gao_exact_zeros++;}gao_record_sign(c,0);return 0;}
    for(var j=i32(count)-1;j>=0;j--){let middle=(prime_table[u32(j)]-1u)/2u;if(gao_digit(u32(j))!=middle){return select(1,-1,gao_digit(u32(j))>middle);}}
    return 1;
}
fn gao_structural_point(v:vec4<u32>,e:u32,i:u32)->vec3<u32>{return signed_bits(coordinates[select(v[i%4u],e,i==4u)].xyz);}
fn gao_negate_bits(x:u32)->u32{return select(x^0x80000000u,0u,(x&0x7fffffffu)==0u);}
fn gao_reflect_bits(q:vec3<u32>,operation:u32)->vec3<u32>{
    var result=q;
    if(operation<3u){result[operation]=gao_negate_bits(q[operation]);}
    else{
        let axis=(operation-3u)%3u;let other=(axis+1u)%3u;result[axis]=q[other];result[other]=q[axis];
        if(operation>=6u){result[axis]=gao_negate_bits(result[axis]);result[other]=gao_negate_bits(result[other]);}
    }
    return result;
}
// Two disjoint pairs reflected across the same plane have parallel chord
// differences. Their lifted differences are also parallel because reflection
// preserves squared norm. Thus four sites are coplanar, and five sites have
// zero insphere determinant, regardless of the unpaired fifth site's position.
fn gao_reflection_pairs(v:vec4<u32>,e:u32,n:u32)->bool{
    for(var operation=0u;operation<9u;operation++){
        var used=0u;var pairs=0u;
        for(var i=0u;i<n;i++){
            if((used&(1u<<i))!=0u){continue;}
            let reflected=gao_reflect_bits(gao_structural_point(v,e,i),operation);
            for(var j=i+1u;j<n;j++){
                if((used&(1u<<j))==0u&&all(reflected==gao_structural_point(v,e,j))){used|=(1u<<i)|(1u<<j);pairs++;break;}
            }
            if(pairs==2u){return true;}
        }
    }
    return false;
}
// Exact reflection pairs cover the dominant grid degeneracies. Keep this
// recognizer small: composing several general recognizers makes Direct3D's
// shader compiler fail after inlining the adaptive predicates. All remaining
// ambiguous cases go through the exact integer determinant.
fn gao_structural_zero(v:vec4<u32>,e:u32,kind:u32)->bool{
    return gao_reflection_pairs(v,e,select(4u,5u,kind==1u));
}
fn gao_numeric_sign(v:vec4<u32>,e:u32,kind:u32)->i32{
    var sign=2i;if(kind==0u){sign=orient_quick(v);}else{sign=sphere_quick(v,e);}
    if(sign==2){sign=double_filter(v,e,kind);}
    if(sign==2){
        if(gao_structural_zero(v,e,kind)){return 0;}
        if(GAO_FILTERED_ONLY){gao_defer_exact=true;return 2;}
        sign=gao_integer_sign(Cache(v,vec4<u32>(e,kind,0u,0u)));
    }
    return sign;
}
// The insphere sign, then on a tie the stable-ID lift perturbation's cofactors (geometry
// is never jittered). Every sign goes through the one gao_numeric_sign call: browsers'
// shader compilers inline each call site, and this body dominates their compile time.
fn gao_inside_exact(ids:vec3<u32>,e:u32)->bool{
    let v=vec4<u32>(ids,gao_seed);if(any(v==vec4<u32>(e))){return false;}
    let original=array<u32,5>(v.x,v.y,v.z,v.w,e);var ordered=original;
    var points=v;var extra=e;var kind=1u;var row=0u;
    for(var rank=0u;rank<6u;rank++){
        if(rank>0u){
            if(rank==1u){for(var i=0u;i<5u;i++){for(var j=i+1u;j<5u;j++){if(all(signed_bits(coordinates[original[i]].xyz)==signed_bits(coordinates[original[j]].xyz))){return false;}}}}
            row=0u;var least=GAO_NONE;
            for(var j=0u;j<5u;j++){if(ordered[j]!=GAO_NONE){let stable=coordinates[ordered[j]].w;if(stable<least){least=stable;row=j;}}}
            var at=0u;for(var j=0u;j<5u;j++){if(j!=row){points[at]=original[j];at++;}}
            extra=GAO_NONE;kind=0u;
        }
        let sign=gao_numeric_sign(points,extra,kind);
        if(sign!=0){if(rank==0u){return sign<0;}return select(sign,-sign,(row&1u)!=0u)<0;}
        if(rank>0u){ordered[row]=GAO_NONE;}
    }
    return false;
}
