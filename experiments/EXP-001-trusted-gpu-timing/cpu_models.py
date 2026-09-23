#!/usr/bin/env python3
"""CPU only conceptual checks, these produce NO GPU performance evidence"""
import math,struct

def f32(x): return struct.unpack("f",struct.pack("f",x))[0]
def fold(xs):
    value=f32(0)
    for x in xs: value=f32(value+f32(x))
    return value
def sectors(stride=1,offset=0):
    # Thirty-two distinct 4-byte accesses; offset/stride are in float elements
    return {((offset+lane*stride)*4)//32 for lane in range(32)}
def tree_stage(xs,block,grid):
    result=[]
    for b in range(grid):
        shared=[]
        for t in range(block):
            values=xs[b*block+t::grid*block]
            shared.append(fold(values))
        stride=block//2
        while stride:
            before=shared.copy()
            for t in range(stride): shared[t]=f32(before[t]+before[t+stride])
            stride//=2
        result.append(shared[0])
    return result
def reduce_model(xs,block=256,sm_count=4):
    if not xs: return 0.0
    while True:
        grid=min((len(xs)+block-1)//block,sm_count*4)
        xs=tree_stage(xs,block,grid)
        if len(xs)==1: return xs[0]

def main():
    for n in [0,1,31,32,33,127,128,129,255,256,257,511,512,513,1003,65537]:
        xs=[(i%17-8)/16 for i in range(n)]
        for block in [128,256,512]:
            assert reduce_model(xs,block)==sum(xs),(n,block)
            if n:
                grid=min((n+block-1)//block,16)
                visits=[0]*n
                for b in range(grid):
                    for t in range(block):
                        for i in range(b*block+t,n,grid*block): visits[i]+=1
                assert all(x==1 for x in visits)
    assert len(sectors(1,0))==4 and len(sectors(1,1))==5
    assert len(sectors(2,0))==8 and len(sectors(8,0))==32
    assert fold([1e8,1,-1e8])==0 and fold([1e8,-1e8,1])==1
    assert math.isclose(12e12/600e9,20)
    assert math.isclose(max(1e6/12e12,12e6/600e9)*1e6,20)
    print("CPU models passed: exact input ownership, dyadic reduction, sectors, rounding, roofline units.")
    print("Float32 order A:",fold([1e8,1,-1e8]),"order B:",fold([1e8,-1e8,1]))
    print("No CUDA compilation, device execution, sanitizer validation or GPU timing has occurred.")

if __name__=="__main__": main()
