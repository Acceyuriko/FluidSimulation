using Unity.Entities;

[GenerateAuthoringComponent]
public struct FluidComponent : IComponentData
{
    public Entity prefab;
}
