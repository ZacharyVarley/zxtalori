// Component-major words coalesce independent stars in the fast phase. A
// cooperative workgroup keeps its entire small workspace in shared memory.
var<workgroup> shared_words:array<u32,32u+2u*FACE_CAP>;
fn word(s:u32,offset:u32)->u32{if(SHARED_FACES){return shared_words[offset];}return scratch[offset*p.seeds+s];}
fn set_word(s:u32,offset:u32,value:u32){if(SHARED_FACES){shared_words[offset]=value;}else{scratch[offset*p.seeds+s]=value;}}
