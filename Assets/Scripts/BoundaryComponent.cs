using Unity.Entities;

[GenerateAuthoringComponent]
public struct BoundaryComponent : IComponentData
{
    public Entity prefab;
}
