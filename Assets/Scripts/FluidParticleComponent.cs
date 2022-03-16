using System;
using Unity.Entities;
using Unity.Mathematics;

[Serializable]
public struct FluidParticleComponent : IComponentData
{
    public Entity Fluid;

    public float radius;
    public float density;
    public float viscosity;
    public float3 gravity;

    public float volume;
}
