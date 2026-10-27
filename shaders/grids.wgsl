// MIT. T/O/I transports ported from SO3K_Transports.
// Lattices are constructed in Zig; every point's transport runs here.
const GRID_PI:f32=3.14159265358979323846;
const GRID_EPS:f32=1e-15;
struct GridParams { count:u32, group:u32, h:u32, pad:u32, beta:f32, p0:f32, p1:f32, p2:f32 }
@group(0) @binding(0) var<uniform> gp:GridParams;
@group(0) @binding(1) var<storage,read> source:array<vec4<f32>>;
@group(0) @binding(2) var<storage,read_write> quaternion:array<vec4<f32>>;

struct Ordered { values:vec3<f32>, indices:vec3<u32> }
fn sort_abs(v:vec3<f32>)->Ordered {
    var a=abs(v); var k=vec3<u32>(0u,1u,2u);
    if(a.x>a.y){let t=a.x;a.x=a.y;a.y=t;let j=k.x;k.x=k.y;k.y=j;}
    if(a.y>a.z){let t=a.y;a.y=a.z;a.z=t;let j=k.y;k.y=k.z;k.z=j;}
    if(a.x>a.y){let t=a.x;a.x=a.y;a.y=t;let j=k.x;k.x=k.y;k.y=j;}
    return Ordered(a,k);
}
fn unorder(v:vec3<f32>, order:vec3<u32>, original:vec3<f32>)->vec3<f32>{
    var r=vec3<f32>(0);r[order.x]=v.x;r[order.y]=v.y;r[order.z]=v.z;
    return r*select(vec3<f32>(-1),vec3<f32>(1),original>=vec3<f32>(0));
}
fn direction(theta:f32,phi:f32)->vec3<f32>{return vec3<f32>(sin(theta)*cos(phi),sin(theta)*sin(phi),cos(theta));}
fn rho3(c:f32)->f32 {let s=max(c,GRID_EPS);return max(0.0,1.5*(atan(1.0/s)-s/(1.0+s*s)));}
fn theta_max(phi:f32)->f32{return atan(1.0/max(sin(phi),GRID_EPS));}
fn t_radius3(theta:f32,phi:f32)->f32 {return rho3(cos(theta)+(cos(phi)+sin(phi))*sin(theta));}
fn t_integral(theta:f32,phi:f32)->f32{
    let half=0.5*theta;var total=0.0;
    for(var j=0u;j<32u;j++){let th=half*(GL32_X[j]+1.0);total+=GL32_W[j]*t_radius3(th,phi)*sin(th);}
    return half*total;
}
fn t_map(cu:vec3<f32>)->vec3<f32>{
    let sorted=sort_abs(cu);let v=sorted.values;let r=length(v);
    let theta=atan2(length(v.xy),v.z);let phi=clamp(atan2(v.y,v.x),GRID_PI/4.0,GRID_PI/2.0);
    let cp=cos(phi);let sp=sin(phi);
    let u=clamp(2.0*sqrt(2.0)*sin(phi-GRID_PI/4.0)/(cp+2.0*sp),0.0,1.0);
    var lo=-1.0;var hi=1.0;var z=clamp(2.0*u-1.0,-1.0,1.0);
    for(var it=0u;it<8u;it++){
        let residual=t_cdf(z)-u;
        if(residual<=0.0){lo=z;}if(residual>=0.0){hi=z;}
        let proposal=z-residual/max(t_density(z)*7.639437268410975,GRID_EPS);
        z=select(0.5*(lo+hi),proposal,proposal>=lo&&proposal<=hi);
    }
    let phi_p=3.0*GRID_PI/8.0+GRID_PI/8.0*clamp(z,-1.0,1.0);
    let y=clamp(pow(sin(theta)*(cp+2.0*sp)/max(cos(theta)+(cp+sp)*sin(theta),GRID_EPS),2.0),0.0,1.0);
    let high=theta_max(phi_p);let area=max(t_density(clamp((phi_p-3.0*GRID_PI/8.0)/(GRID_PI/8.0),-1.0,1.0)),GRID_EPS);
    var th=2.0*asin(clamp(sqrt(y)*sin(0.5*high),0.0,1.0));
    for(var it=0u;it<12u;it++){
        th=clamp(th,0.0,high);let cumulative=t_integral(th,phi_p);
        let slope=max(t_radius3(th,phi_p)*sin(th)/area,GRID_EPS);
        let step=(cumulative/area-y)/slope;
        th=clamp(th-select(step,0.25*step,abs(step)>0.25),0.0,high);
    }
    let r_src=0.8512554803691886/max(cos(theta)+sin(theta)*(cp+sp),GRID_EPS);
    let r_tar=pow(t_radius3(th,phi_p),1.0/3.0)*(r/r_src);
    return unorder(r_tar*direction(th,phi_p),sorted.indices,cu);
}

fn o_cheb(t:f32,which:u32)->vec2<f32>{
    if(which==0u){return vec2<f32>(o_a_left(t),o_a_left_d(t));}
    if(which==1u){return vec2<f32>(o_b_left(t),o_b_left_d(t));}
    if(which==2u){return vec2<f32>(o_a_right(t),o_a_right_d(t));}
    return vec2<f32>(o_b_right(t),o_b_right_d(t));
}
fn o_phi(u:f32)->vec2<f32>{
    let left=u<=O_UBREAK;let low=select(O_UBREAK,0.0,left);let high=select(1.0,O_UBREAK,left);
    let t=2.0*(u-low)/(high-low)-1.0;
    let p=o_cheb(t,select(2u,0u,left));let q0=o_cheb(t,select(3u,1u,left));let q=1.0+q0.x;
    return vec2<f32>(clamp(p.x/q,GRID_PI/4.0,GRID_PI/2.0),max((2.0/(high-low))*(p.y*q-p.x*q0.y)/(q*q),GRID_EPS));
}
fn o_source_azimuth(phi:f32,beta:f32)->f32{
    let phib=atan(1.0/(beta-1.0));let ct=0.5-pow(2.0-beta,3.0)/12.0;let ub=(ct-0.5*(beta-1.0))/ct;
    if(phi>phib){return clamp(ub+(1.0-ub)*clamp(1.0-cos(phi)/(max(sin(phi),GRID_EPS)*(beta-1.0)),0.0,1.0),0.0,1.0);}
    let t=clamp(2.0*(phi-GRID_PI/4.0)/(phib-GRID_PI/4.0)-1.0,-1.0,1.0);
    let poly=o_source_poly(t,beta-sqrt(2.0));
    return clamp(ub*clamp(0.5*(1.0+t)+(1.0-t*t)*poly,0.0,1.0),0.0,1.0);
}
fn o_fsum(t:f32,s:f32)->f32{let q=1.0+s*t;return (-1.0/q+0.5/(q*q))/(s*s);}
fn o_gsource(theta:f32,phi:f32,beta:f32)->f32{
    let s=max(cos(phi)+sin(phi),GRID_EPS);let ths=atan(beta/s);let tt=tan(theta);let ts=tan(ths);
    if(theta<=ths){return 0.5*tt*tt;}
    return max(0.0,0.5*ts*ts+pow(1.0+beta,3.0)*(o_fsum(tt,s)-o_fsum(ts,s)));
}
fn o_radius3(theta:f32,phi:f32)->f32{return rho3(max((1.0+sqrt(2.0))*cos(theta),cos(theta)+sin(theta)*(cos(phi)+sin(phi))));}
fn o_integral_piece(low:f32,high:f32,phi:f32,cut:bool)->f32{
    let half=0.5*(high-low);let mid=0.5*(high+low);var total=0.0;
    for(var j=0u;j<16u;j++){
        let th=half*O_GL16_X[j]+mid;var c=(1.0+sqrt(2.0))*cos(th);
        if(cut){c=cos(th)+sin(th)*(cos(phi)+sin(phi));}
        total+=O_GL16_W[j]*rho3(c)*sin(th);
    }
    return half*total;
}
fn o_integral(theta:f32,phi:f32)->f32{
    let split=atan(sqrt(2.0)/max(cos(phi)+sin(phi),GRID_EPS));
    if(theta<=split){return o_integral_piece(0.0,theta,phi,false);}
    return o_integral_piece(0.0,split,phi,false)+o_integral_piece(split,theta,phi,true);
}
fn o_map(cu:vec3<f32>)->vec3<f32>{
    let sorted=sort_abs(cu);let v=sorted.values;let r=length(v);
    let theta=atan2(length(v.xy),v.z);let phi=atan2(v.y,v.x);
    let mapped_target=o_phi(o_source_azimuth(phi,gp.beta));let phi_p=mapped_target.x;let area=max((GRID_PI*GRID_PI/384.0)/mapped_target.y,GRID_EPS);
    let y=clamp(o_gsource(theta,phi,gp.beta)/max(o_gsource(theta_max(phi),phi,gp.beta),GRID_EPS),0.0,1.0);
    let high=theta_max(phi_p);var th=2.0*asin(clamp(sqrt(y)*sin(0.5*high),0.0,1.0));
    for(var it=0u;it<8u;it++){
        th=clamp(th,0.0,high);let gn=clamp(o_integral(th,phi_p)/area,0.0,1.0);
        let slope=max(o_radius3(th,phi_p)*sin(th)/area,GRID_EPS);let step=(gn-y)/slope;
        th=clamp(th-select(step,0.25*step,abs(step)>0.25),0.0,high);
    }
    let r_src=1.0725146985555127*min(1.0/max(cos(theta),GRID_EPS),(1.0+gp.beta)/max(cos(theta)+sin(theta)*(cos(phi)+sin(phi)),GRID_EPS));
    let r_tar=pow(o_radius3(th,phi_p),1.0/3.0)*(r/r_src);
    return unorder(r_tar*direction(th,phi_p),sorted.indices,cu);
}

fn cube_to_ball(cu:vec3<f32>)->vec3<f32>{
    let sorted=sort_abs(cu);let a=sorted.values;let small=a.x;let mid=a.y;let big=a.z;
    if(big==0.0){return vec3<f32>(0);}
    let arg=select(0.0,small*(GRID_PI/12.0)/max(mid,GRID_EPS),mid>0.0);let c=cos(arg);let s=sin(arg);let gap=min(c-sqrt(2.0),-GRID_EPS);
    let rad=(4.0*big*big*gap+sqrt(2.0)*mid*mid*(3.0-2.0*sqrt(2.0)*c))/gap;
    let face=pow(2.0,1.0/12.0)*pow(3.0,1.0/3.0)*mid*sqrt(max(rad,0.0))/(pow(GRID_PI,1.0/3.0)*big*sqrt(sqrt(2.0)-c));
    let x=face*s;let y=face*(sqrt(2.0)*c-1.0)/sqrt(2.0);
    let z=(2.0*pow(6.0,1.0/3.0)*big*big*gap+pow(2.0,5.0/6.0)*pow(3.0,1.0/3.0)*mid*mid*(3.0-2.0*sqrt(2.0)*c))/(2.0*pow(GRID_PI,1.0/3.0)*big*gap);
    return unorder(vec3<f32>(x,y,z),sorted.indices,cu);
}
fn i_theta_high(phi:f32)->f32{
    var maximum=0.0;
    for(var j=0u;j<4u;j++){let p=I_PLANES[j];maximum=max(maximum,atan2(p.z,p.x*cos(phi)+p.y*sin(phi)));}
    return GRID_PI-maximum;
}
fn i_radius3(theta:f32,phi:f32)->f32{return rho3(dot(direction(theta,phi),I_FACE));}
fn i_integral(theta:f32,phi:f32)->f32{
    let half=0.5*theta;var sum=0.0;
    for(var j=0u;j<32u;j++){let th=half*(1.0+GL32_X[j]);sum+=GL32_W[j]*i_radius3(th,phi)*sin(th);}
    return half*sum;
}
fn i_map(cu:vec3<f32>)->vec3<f32>{
    let ho=cube_to_ball(cu);let radius=length(ho);if(radius==0.0){return ho;}
    let d=ho/radius;var best=0u;var score=dot(d,I_SITES[0]);
    for(var j=1u;j<120u;j++){let next=dot(d,I_SITES[j]);if(next>score){score=next;best=j;}}
    let folded=I_OPS[best]*d;let theta=atan2(length(folded.xy),folded.z);let phi=atan2(folded.y,folded.x);
    let hi=i_theta_high(phi);let y=clamp(pow(sin(0.5*theta)/max(sin(0.5*hi),GRID_EPS),2.0),0.0,1.0);
    let u=clamp(i_source(clamp(phi/(GRID_PI/10.0)-1.0,-1.0,1.0)),0.0,1.0);
    let phi_p=clamp(i_azimuth(2.0*u-1.0),0.0,GRID_PI/5.0);let high=i_theta_high(phi_p);let area=max(i_integral(high,phi_p),GRID_EPS);
    var low=0.0;var upper=high;var th=2.0*asin(clamp(sqrt(y)*sin(0.5*high),0.0,1.0));
    for(var it=0u;it<16u;it++){
        let residual=i_integral(th,phi_p)/area-y;
        if(residual<0.0){low=th;}if(residual>0.0){upper=th;}
        let slope=max(i_radius3(th,phi_p)*sin(th)/area,GRID_EPS);let proposal=th-residual/slope;
        th=select(proposal,0.5*(low+upper),proposal<low||proposal>upper);
    }
    let mapped_target=direction(th,phi_p);let mapped=mapped_target*(radius*pow(i_radius3(th,phi_p),1.0/3.0)/1.330670039491469);
    return transpose(I_OPS[best])*mapped;
}
fn omega_minus_sin(w:f32)->f32{
    let s=w*w;
    if(w<1.0){return w*s/6.0*(1.0+s*(-0.05+s*(1.0/840.0+s*(-1.0/60480.0+s*(1.0/6652800.0+s*(-1.0/1037836800.0+s/219419136000.0))))));}
    return w-sin(w);
}
fn ho_quaternion(ho:vec3<f32>)->vec4<f32>{
    let rho=length(ho);if(rho==0.0){return vec4<f32>(1,0,0,0);}
    var w=2.0*rho;
    for(var it=0u;it<6u;it++){
        let rw=pow(max(0.75*omega_minus_sin(w),0.0),1.0/3.0);
        let sh=sin(0.5*w);let slope=2.0*sh*sh/max(4.0*rw*rw,GRID_EPS);
        w=clamp(w-(rw-rho)/max(slope,GRID_EPS),0.0,GRID_PI);
    }
    return normalize(vec4<f32>(cos(0.5*w),ho*(sin(0.5*w)/rho)));
}
@compute @workgroup_size(64)
fn grid_generate(@builtin(global_invocation_id) id:vec3<u32>){
    let i=id.x;if(i>=gp.count){return;}let cu=source[i].xyz;
    if(dot(cu,cu)==0.0){quaternion[i]=vec4<f32>(1,0,0,0);return;}
    var ho=vec3<f32>(0);if(gp.group==8u){ho=t_map(cu);}else if(gp.group==9u){ho=o_map(cu);}else{ho=i_map(cu);}
    quaternion[i]=ho_quaternion(ho);
}
