using Unity.Entities;

[GenerateAuthoringComponent]
public struct BoundaryParticleComponent : IComponentData
{
    public float radius;
    public float mass;
}
