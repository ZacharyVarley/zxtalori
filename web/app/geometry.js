// Plot geometry only. Orientation generation and optimization remain in Zig/WGSL.
export function multiply(a,b){return [a[0]*b[0]-a[1]*b[1]-a[2]*b[2]-a[3]*b[3],a[0]*b[1]+b[0]*a[1]+a[2]*b[3]-a[3]*b[2],a[0]*b[2]+b[0]*a[2]+a[3]*b[1]-a[1]*b[3],a[0]*b[3]+b[0]*a[3]+a[1]*b[2]-a[2]*b[1]];}
const dot=(a,b)=>a.reduce((s,x,i)=>s+x*b[i],0);
const cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
export function project(q,projection){let [w,x,y,z]=q;const norm=Math.hypot(w,x,y,z);if(!norm)return null;const sign=w<0?-1:1;w=w*sign/norm;x=x*sign/norm;y=y*sign/norm;z=z*sign/norm;const s=Math.hypot(x,y,z);if(s<1e-12)return [0,0,0];if(projection==='rodrigues')return w<1e-9?null:[x/w,y/w,z/w];const angle=2*Math.atan2(s,w),r=projection==='axisangle'?angle:Math.cbrt(.75*(angle-Math.sin(angle)));return [x*r/s,y*r/s,z*r/s];}
function fromRodrigues(r,projection){if(projection==='rodrigues')return r;const n=Math.hypot(...r);if(n===0)return [0,0,0];const angle=2*Math.atan(n),radius=projection==='axisangle'?angle:Math.cbrt(.75*(angle-Math.sin(angle)));return r.map(x=>x*radius/n);}
export function polyhedron(ops,group){
  const planes=[];
  for(let k=4;k<ops.length;k+=4){const w=ops[k],a=Math.abs(w);if(group===8&&Math.abs(a-.5)>1e-5||group===9&&a<.49||group===10&&Math.abs(a-.809016994375)>1e-5)continue;
    for(const sign of [-1,1]){const n=Array.from(ops.subarray(k+1,k+4),x=>x*sign),length=Math.hypot(...n);if(length<1e-9)continue;const normal=n.map(x=>x/length),offset=(1-sign*w)/length;if(!planes.some(p=>Math.hypot(...p.n.map((x,i)=>x-normal[i]))<1e-6&&Math.abs(p.d-offset)<1e-6))planes.push({n:normal,d:offset});}
  }
  const vertices=[];
  for(let i=0;i<planes.length;i++)for(let j=i+1;j<planes.length;j++)for(let k=j+1;k<planes.length;k++){
    const a=planes[i],b=planes[j],c=planes[k],bc=cross(b.n,c.n),den=dot(a.n,bc);if(Math.abs(den)<1e-9)continue;const ca=cross(c.n,a.n),ab=cross(a.n,b.n),p=bc.map((x,l)=>(x*a.d+ca[l]*b.d+ab[l]*c.d)/den);
    if(planes.some(f=>dot(f.n,p)>f.d+2e-6)||vertices.some(v=>Math.hypot(...v.p.map((x,l)=>x-p[l]))<1e-5))continue;
    vertices.push({p,faces:planes.flatMap((f,l)=>Math.abs(dot(f.n,p)-f.d)<2e-6?[l]:[])});
  }
  const edges=[];for(let i=0;i<vertices.length;i++)for(let j=i+1;j<vertices.length;j++)if(vertices[i].faces.filter(f=>vertices[j].faces.includes(f)).length>=2)edges.push([vertices[i].p,vertices[j].p]);
  return {vertices:vertices.map(v=>v.p),edges};
}
export function boundary(ops,group,projection,extended=false,clip=4){
  const wire=[];let extent=0;
  if(!extended&&group>=4){const poly=polyhedron(ops,group);for(const [a,b] of poly.edges){const line=[];for(let i=0;i<=40;i++)line.push(fromRodrigues(a.map((x,k)=>x+(b[k]-x)*i/40),projection));wire.push(line);}extent=Math.max(...poly.vertices.map(p=>Math.hypot(...fromRodrigues(p,projection))),.01);return {wire,extent,edges:poly.edges.length,vertices:poly.vertices.length};}
  const radial=dir=>{let t=Infinity;if(!extended)for(let k=4;k<ops.length;k+=4){const u=dot(dir,ops.subarray(k+1,k+4));if(Math.abs(u)>1e-12)t=Math.min(t,(1-Math.sign(u)*ops[k])/Math.abs(u));}const angle=2*Math.atan(t),radius=projection==='rodrigues'?Math.min(clip,t):projection==='axisangle'?angle:Math.cbrt(.75*(angle-Math.sin(angle)));return dir.map(x=>x*radius);};
  for(let ring=1;ring<12;ring++){const theta=ring*Math.PI/12,line=[];for(let k=0;k<=96;k++){const phi=2*Math.PI*k/96;line.push(radial([Math.sin(theta)*Math.cos(phi),Math.sin(theta)*Math.sin(phi),Math.cos(theta)]));}wire.push(line);}
  for(let k=0;k<16;k++){const phi=2*Math.PI*k/16,line=[];for(let ring=0;ring<=96;ring++){const theta=ring*Math.PI/96;line.push(radial([Math.sin(theta)*Math.cos(phi),Math.sin(theta)*Math.sin(phi),Math.cos(theta)]));}wire.push(line);}
  // The cyclic FZ reaches pi at its equator. SO(3) has the same outer radius.
  extent=projection==='rodrigues'?clip:projection==='axisangle'?Math.PI:Math.cbrt(.75*Math.PI);
  return {wire,extent,edges:null,vertices:null};
}

// Generate only the requested display images. Never allocate the extended cloud.
// Points come back packed for the renderer: positions xyz, then each one's sample index
// and symmetry operation.
export function displayPoints(q,ops,projection,extended,limit,clip=Infinity){
  if(!q.length||!ops.length)return {xyz:new Float32Array(),index:new Int32Array(),operation:new Uint8Array(),count:0,total:0,omitted:0};
  const order=extended?ops.length/4:1,total=q.length/4*order,count=limit?Math.min(limit,total):total;let kept=0,omitted=0;
  const xyz=new Float32Array(count*3),indices=new Int32Array(count),operations=new Uint8Array(count);
  const gcd=(a,b)=>b?gcd(b,a%b):a;let stride=Math.max(1,Math.floor(total/count));while(extended&&gcd(stride,total)!==1)stride++;
  // A coprime stride avoids locking a display subset to one symmetry operator
  // when the display count happens to divide N|K| (e.g. I, 60,000 -> 1,000).
  for(let k=0;k<count;k++){const image=extended?k*stride%total:Math.floor(k*total/count),index=Math.floor(image/order),operation=image%order;const a=q.subarray(index*4,index*4+4),p=project(extended?multiply(a,ops.subarray(operation*4,operation*4+4)):a,projection);if(p&&Math.hypot(p[0],p[1],p[2])<=clip*(1+1e-7)){xyz.set(p,kept*3);indices[kept]=index;operations[kept]=operation;kept++;}else omitted++;}
  return {xyz:xyz.subarray(0,kept*3),index:indices.subarray(0,kept),operation:operations.subarray(0,kept),count:kept,total,omitted};
}
