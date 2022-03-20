using System;
using Unity.Entities;

[GenerateAuthoringComponent]
[Serializable]
public struct FluidParticleComponent : IComponentData
{
    public float radius;
    public float restDensity;
    public float viscosity;
    public float gasConstant;
}
