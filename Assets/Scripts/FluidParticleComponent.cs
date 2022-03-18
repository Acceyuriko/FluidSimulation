using System;
using Unity.Entities;

[GenerateAuthoringComponent]
[Serializable]
public struct FluidParticleComponent : IComponentData
{
    public Entity Fluid;

    public float radius;
    public float density;
    public float mass;
    public float viscosity;
}
