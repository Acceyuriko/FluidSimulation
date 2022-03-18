using System;
using Unity.Entities;

[GenerateAuthoringComponent]
[Serializable]
public struct FluidParticleComponent : IComponentData
{
    public float radius;
    public float density;
    public float viscosity;
    public float gasConstant;
}
