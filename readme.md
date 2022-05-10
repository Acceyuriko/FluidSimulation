## Buoyancy and Fluid Simulation
[中文介绍](readme_zh.md)

![preview](./Preview/preview.gif)

## Preface

There was a project that needed to implement a buoyancy system,
so I did some research.
However, I am not a professional learner of fluid mechanics.
If there are any mistakes, please point them out.

In general, buoyancy can be achieved in two ways.
- Use a rigid body to simulate the position of the fluid,
calculate the part where the buoyancy object intersects the fluid,
calculate the buoyancy,
and apply it to the buoyancy object, such as [https://youtu.be/iasDPyC0QOg](https://youtu.be/iasDPyC0QOg)
    - This method is simple to implement and suitable for large scenes where the fluid is relatively static,
    such as oceans and lakes

- Simulate fluid behavior with a large number of particles.
When other objects collide with fluid particles,
the physics engine can complete the buoyancy calculation.
Such as [https://www.youtube.com/watch?v=tzc0Iq9Zgt4](https://www.youtube.com/watch?v=tzc0Iq9Zgt4)
    - This method is more complex to implement,
    needs to solve the behavior equations of fluid particles,
    and consumes more computing resources.
    It is suitable for small scenes of fluid motion, such as water cups and ocean waves.

The second method is mainly discussed here.

## Common method
The Lagrangian perspective and the Euler perspective are
two common methods used to study the behavior of fluids and deformable solids.

The Lagrangian perspective is particle-based.
Fluid data, such as density, mass, and velocity, are stored in particles,
which changes position as the simulation proceeds.
Each particle can be regarded as a fluid micelle,
and the Lagrangian method simulates the motion caused by interaction between particles.

This is a method of solving the NS equations that **discretizes the matter itself**.
This method has many advantages, such as intuitive understanding, energy conservation.
However, there are also points that the Lagrangian perspective is not good at.
The biggest one is the neighbor search of particles.
The time to find neighbors is generally the bottleneck of the algorithm.

The Euler perspective is grid-based,
and records fluid data at a fixed set of space points.
The meaning of the attribute on this space point indicates how much the attribute passes through this space point.
The position of the space point will not change as the simulation proceeds.

This is a method of solving NS equations that **discretizes the space**.
The advantage of this method is that the data of the entire field is naturally discretized to the space points,
and the data of the surrounding space points can be obtained by adding a fixed offset to the space point.
But the Euler perspective often suffers from energy dissipation when processing motion of particles.

In practice, compared with the grid-based method,
the particle-based method has the advantages of energy conservation and no boundary region,
and it is easier to simulate complex phenomena,
such as rolling waves, water droplets, fluid and solid motion, etc.
Therefore, the Lagrangian perspective is the most widely used fluid simulation method.

`Smoothed Particle Hydrodynamics` and its variant `Position Based Dynammics` are two common Lagrangian perspective approach.

## References
- [Unity ECS Job System SPH](https://github.com/leonardo-montes/Unity-ECS-Job-System-SPH)
- [how to implement a fluid simulation on the cpu with unity ecs job system](https://medium.com/@leomontes_60748/how-to-implement-a-fluid-simulation-on-the-cpu-with-unity-ecs-job-system-bf90a0f2724f)
- [gentle introduction to fluid simulation for programmers and technical artists](https://shahriyarshahrabi.medium.com/gentle-introduction-to-fluid-simulation-for-programmers-and-technical-artists-7c0045c40bac)
- [SPH流体模拟基础](https://zhuanlan.zhihu.com/p/363054480)
- [2014_EG_SPH_STAR.pdf](https://cg.informatik.uni-freiburg.de/publications/2014_EG_SPH_STAR.pdf)
- [Implementing Bitonic Merge Sort in Vulkan Compute](https://poniesandlight.co.uk/reflect/bitonic_merge_sort/)

