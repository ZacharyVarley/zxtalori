// Covering checks have a different cost from pair relaxation. Estimate the next
// check from measured warm latency; keep expensive automatic work out of a run.
export function coveringEstimate(n,order,model){
  if(model?.milliseconds>0&&!model.cold)return model.milliseconds*(n/model.count)**1.15*(order/model.order)**.3;
  return 40*(n/1000)**1.15*(order/24)**.3;
}
export function shouldCheckCovering({mode,phase,count,order,age,elapsed,stepMilliseconds,model,failed=false}){
  if(mode==='manual'||failed)return false;
  if(mode==='end')return phase==='end'&&age>0;
  if(mode==='ten')return phase==='generation'||phase==='end'&&age>0||phase==='step'&&age>=10;
  const cost=coveringEstimate(count,order,model);
  const warm=model?.milliseconds>0&&!model.cold;
  if(count>10000&&!warm||cost>1200)return false;
  if(phase==='generation')return true;
  if(phase==='end')return age>0;
  if(cost>400)return false;
  return age>=5&&elapsed>=Math.max(600,cost*5)&&age*Math.max(1,stepMilliseconds)>=cost*3;
}
