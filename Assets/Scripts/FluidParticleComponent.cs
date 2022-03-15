using System;
using Unity.Entities;
using Unity.Mathematics;

[Serializable]
public struct FluidParticleComponent : ISharedComponentData
{
    public Entity Fluid;

    public float radius;
    public float density;
    public float3 gravity;

    public float volume;
}
