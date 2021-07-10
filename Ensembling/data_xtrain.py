// UNSIGNED !!!!
float plane(float3 p)
{
return abs(p.y);
}


float sphere(float3 p,float x)
{
return length(p)-x;
}


float cube(float3 p,float3 x)
{
//return max(max(abs(p.x)-x.x,abs(p.y)-x.y),abs(p.z)-x.z);	// many asciis
//return length(max(abs(p)-x,0.));							// unsigned, artefacts
	p=abs(p)-x;					// smallest in
return max(max(p.x,p.y),p.z);	// ascii-count -> 42 :D
}

// UNSIGNED !!!!
float rcube(float3 p,float3 x,float y )
{
return length(max(abs(p)-x,0.))-y;
}
// SIGNED - TEST
float srcube(float3 p,float3 x,float y )
{
//return max(max(abs(p.x)-x.x-y,abs(p.y)-x.y-y),abs(p.z)-x.z-y);
	p=abs(p)-(x-y);					// smallest in
return max(max(p.x,p.y),p.z);	// ascii-count -> 42 :D
}


float ring(float3 p,float x,float y,float z)
{
//return max(abs(length(p.xz)-r)-r2,abs(p.y)-c);
return max(abs(length(p.xy)-x)-y,abs(p.z)-z);
}


float octahedron(float3 p,float x)
{
   p=abs(p);
   return (p.x+p.y+p.z-x)/3;
}
// TEST ONLY!
float octahedron_extruded(float3 p,float x)
{
//   p=abs(clamp(p,-1.,1.));
   p=clamp(abs(p),0.,1.);
   return (p.x+p.y+p.z-x)/3;
}


float cylinderX(float3 p,float x,float y)
{
	return max(abs(p.x)-y,length(p.yz)-x);
}
float cylinderY(float3 p,float x,float y)
{
	return max(abs(p.y)-y,length(p.xz)-x);
}
float cylinderZ(float3 p,float x,float y)
{
	return max(abs(p.z)-y,length(p.xy)-x);
//	return length(p.xy-y)-x;
}


float torusX(float3 p,float x,float y)
{
return length(float2(length(float2(p.y,p.z))-x,p.x))-y;
}
float torusY(float3 p,float x,float y)
{
return length(float2(length(float2(p.x,p.z))-x,p.y))-y;
}
float torusZ(float3 p,float x,float y)
{
return length(float2(length(float2(p.x,p.y))-x,p.z))-y;
}


float hexagonX(float3 p,float x,float y)
{
	p=abs(p);
return max(p.x-y,max(p.y+p.z*.5,p.z)-x);
}
float hexagonY(float3 p,float x,float y)
{
	p=abs(p);
return max(p.y-y,max(p.z+p.x*.5,p.x)-x);
}
float hexagonZ(float3 p,float x,float y)
{
	p=abs(p);
//return max(p.z-y,max(p.x+p.y*0.57735,p.y)-x);
return max(p.z-y,max(p.x+p.y*.5,p.y)-x);
}


float octagonX(float3 p,float x,float y)
{
	p=abs(p);
return max(p.x-y,max(p.z+p.y*.5,p.y+p.z*.5)-x);
}
float octagonY(float3 p,float x,float y)
{
	p=abs(p);
return max(p.y-y,max(p.z+p.x*.5,p.x+p.z*.5)-x);
}
float octagonZ(float3 p,float x,float y)
{
	p=abs(p);
return max(p.z-y,max(p.y+p.x*.5,p.x+p.y*.5)-x);
}


float capsuleX(float3 p,float x,float y)
{
	p=abs(p);
return min(max(p.x-x,length(p.yz)-y),length(p-float3(x,0.,0.))-y);
}
float capsuleY(float3 p,float x,float y)
{
	p=abs(p);
return min(max(p.y-x,length(p.xz)-y),length(p-float3(0.,x,0.))-y);
}
float capsuleZ(float3 p,float x,float y)
{
	p=abs(p);
return min(max(p.z-x,length(p.xy)-y),length(p-float3(0.,0.,x))-y);
}


float prismX(float3 p,float x,float y)
{
return max(abs(p.z)-y,max(abs(p.y)*.9+p.x*.5,-p.x)-x*.5);
}
float prismY(float3 p,float x,float y)
{
return max(abs(p.z)-y,max(abs(p.x)*.9+p.y*.5,-p.y)-x*.5);
}
float prismZ(float3 p,float x,float y)
{
return max(abs(p.y)-y,max(abs(p.x)*.9+p.z*.5,-p.z)-x*.5);
}


// Strange Stuff:

// y=.5 ??
float eightspheres(float3 p,float3 x,float y )
{
return length(abs(p)-x)-y;
}

// y=.5
float wrongHexahedron(float3 p,float3 x,float y )
{
	p=abs(p)-x;
	p+=y*p.zxy;
return max(max(p.x,p.y),p.z);
}

// x,y = .5
float wronglyOctagon(float3 p,float3 x,float y )
{
	p=abs(p)-x;
	p+=y*p.zyx;
return max(max(p.x,p.y),p.z);
}