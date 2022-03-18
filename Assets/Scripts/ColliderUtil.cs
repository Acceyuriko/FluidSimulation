using Unity.Physics;

public static class ColliderUtil
{
    unsafe public static float GetColliderRadius(PhysicsCollider collider)
    {

        var ptr = (SphereCollider*)collider.ColliderPtr;
        return ptr->Radius;
    }
}