---
layout: page
title: Research
permalink: /research/
description: The summary of my projects.
nav: true
nav_order: 2
display_categories: 
horizontal: false
---

## Physics-based Modeling and Differentiable Simulation
###### Representive Papers : [[Deformable Rope, RA-L'23]](https://entongsu.github.io/differential_rope-github.io/), [[Soft Tissue, ICRA'21]](https://ieeexplore.ieee.org/document/9561177), [[Soft Tissue, ICRA'24]](https://arxiv.org/abs/2309.11656), [[Surgical Thread, ICRA'23]](https://ieeexplore.ieee.org/abstract/document/10161539)
The focus of this work is on the **modeling and simulation of robotic scenes in a uniformed representation**. Real-time robotic applications encompass a diverse range of materials and tools, including soft, deformable, fluid objects, and rigid, articulated, or soft tools. The challenge of addressing this diversity is met by leveraging computer graphics methods to manage different material properties, kinematics, and dynamics. The chosen approach employs the position-based dynamics (PBD) method to efficiently simulate and model various objects in real-time, incorporating particle dynamics with physical and geometric constraints. This formulation offers the advantages of rapid positional adjustment and stable convergence in parallel. Simultaneously, all constraints can be defined in a differentiable manner, establishing a strong connection to data-driven methods (e.g., neural network).

The unified simulation and modeling approach has been applied, particularly in surgical contexts. This application extends to surgical tools such as da Vinci Research Kit robotic tools, deformable volumes, soft tissue, and viscous fluids like blood. In the context of surgical navigation and manipulation tasks, particle-based tool-tissue interaction modeling with 2D mesh structures has been considered. For general robotics applications, various structural objects with different types of geometrical constraints have been modeled to capture physical effects. For instance, in rope manipulation, universal particle-based modeling with serial-linked robotic manipulators has been employed. Safety constraints have also been taken into account for cloth trajectory optimization. Additionally, geometrical models like Bézier curves for soft robots and polynomial splines for surgical threads have been used, both of which can be easily formalized into curve-based constraints with discrete sets of particles representing thin structural objects. In summary, the proposed physics-based modeling approach can represent a broad spectrum of multi-stiffness objects and tools for dexterous manipulation, as well as robot modeling from rigid to soft structures.

<table width="920">
	<tr>
		<td width="460">
			<iframe width="460" height="258" src="https://www.youtube.com/embed/hTN-ttMhpcQ" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen=""></iframe>
		</td>
		<td width="460">
			<iframe width="460" height="258" src="https://www.youtube.com/embed/DxUhwZvWw-k" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen=""></iframe>
		</td>
	</tr>
  <tr>
		<td width="460">
			<iframe width="460" height="258" src="https://www.youtube.com/embed/fVEAxvItFmw" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen=""></iframe>
		</td>
		<td width="460">
			<iframe width="460" height="258" src="https://www.youtube.com/embed/94s5siIJdGk" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen=""></iframe>
		</td>
	</tr>
</table>

<br/><br/>

## Advanced Planning, Control and Learning
###### Representive Papers : [[Articulated Robots, T-RO'23]](https://entongsu.github.io/differential_rigid-github.io/), [[Blood Suction, RA-L'21]](https://ieeexplore.ieee.org/document/9561624), [[Blood Suction, ICRA'21]](https://ieeexplore.ieee.org/document/9561624), [[Cloth Safety Control, ICRA'24]](https://arxiv.org/abs/2309.11655)
The characterization of **advanced planning, control and learning modules for integration into the robotic system** is another crucial area of my past research. In particular, my recently developed positional constraints-based multi-body dynamics (PBD) framework can solve state-of-the-art control problems. More importantly, it supports arbitrary differentiation for an objective function with respect to all physical variables involving control input, system states, initial conditions and parameters etc. Therefore, many robotic inverse problems (such as parameter identification, motion control and planning, and trajectory optimization) can be formulated into constrained minimization solutions where the state variables are subject to equality and inequality constraints derived from physical principles. 

Thanks to the differentiable property with ready-to-use gradients using our PBD modeling approach, several downstream robotic applications are carried out, including impedance control and trajectory planning of compliant serial elastic Baxter arms, shape control of linear-deformable objects, trajectory optimization of cloth, and blood fluid with trajectory planning and model-predictive control (MPC). More significantly, our framework can be transferred to *real-to-sim/sim-to-real* experimental setups, closing the "gap" between the simulated systems from real-world cases. I've made an attempt to build an online, continuous, *real-to-sim* registration method for 3D visual perception (surface point cloud) and physics-based PBD simulation (volumetric mesh). The application is soft tissue manipulation using da Vinci surgical tools. To summarize, my work has a high potential for efficiently solving model-based or data-driven control/planning problems with a differentiable optimization formulation. Additionally, many control policies can be trained in an end-to-end fashion using learning approaches, such as our work on 4D lung motion prediction, scene reconstruction.

<table width="920">
	<tr>
		<td width="460">
			<iframe width="460" height="258" src="https://www.youtube.com/embed/RTl-egsjKvM" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen=""></iframe>
		</td>
		<td width="460">
			<iframe width="460" height="258" src="https://www.youtube.com/embed/ytIRd9gfiQE" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen=""></iframe>
		</td>
	</tr>
</table>

<br/><br/>

## Integrated Robotic System and Autonomy
###### Representive Papers : [[Haptics, Robotica]](https://www.cambridge.org/core/journals/robotica/article/an-energybased-approach-for-ndof-passive-dualuser-haptic-training-systems/80D42C872561E397B90F160ACD65A2F3), [[Autonomy Review, Applied Sciences'21]](https://www.mdpi.com/2076-3417/11/1/209), [[Navigation, IROS'20]](https://ieeexplore.ieee.org/document/9341283), [[Underwater Robot]](https://drive.google.com/file/d/1WP7PegDWiNI6Glp-0dRbhA1Gj9ogs_-G/edit)
Building upon prior contributions in modeling and control, a focus has been placed on **integrating system autonomy across various hardware-specific robotic platforms**. The autonomy levels, as outlined in review papers on robotics, encompass teleoperation control (low-level), shared control (middle-level), trade control (middle-level), supervisory control (middle-level), and fully autonomous control (high-level). Despite variations in the mechanical designs of classical articulated robots, differing in structure and pattern, diverse autonomy levels for each robot can consistently be created by adhering to the modeling of kinematics and dynamics, as well as planning and control. Relevant contributions can be identified in autonomous navigation for a non-holonomic mobile robot, navigation control of a low-cost quad-rotor robot, robotic projectile launch tasks, and teleoperation control for a hydraulic-driven underwater robotic arm.

Recent years have witnessed the development of advanced robotic designs for continuum robots, soft robots, and haptic devices. Continuum robots, with their adaptive mechanism and more dexterous pattern, have emerged as a viable alternative to traditional rigid-link manipulators. The introduction of soft material fabrication for continuum robots facilitates environment-compliant interaction. Addressing the kinematic redundancy of continuum soft robots for autonomous planning and navigation, the shape reconstruction method developed is crucial for precise and reliable motion control, requiring only monocular images and no additional sensors.

Haptics, as another critical area for shared autonomy (in middle-level), aims to provide force/torque feedback to the human user. This is particularly relevant for enhancing operational safety in collaborative and unstructured workspaces, such as in surgical robots. The Ph.D. Dissertation focused on developing a dual-user haptic system dedicated to medical training usage. Haptics and shared control approaches were employed to establish a safe human-robot interaction framework. Utilizing an energetic approach (port-Hamiltonian modeling), the system architecture was extended to account for distributed time delays. In summary, autonomy has been implemented in both traditional and innovatively designed robots through hardware-specific modeling and control.

<table width="920">
	<tr>
		<td width="460">
			<iframe width="460" height="258" src="https://www.youtube.com/embed/k4-E4F91VGk" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen=""></iframe>
		</td>
		<td width="460">
			<iframe width="460" height="258" src="https://www.youtube.com/embed/M18yemz-KhQ" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen=""></iframe>
		</td>
	</tr>
</table>

<br/><br/>

## Computer Vision based Perception and Reconstruction
###### Representive Papers : [[4D Lung Reconstrcution, T-MBE'23]](https://ieeexplore.ieee.org/document/10144816), [[Robot Reconstrcution, ICRA'23]](https://ieeexplore.ieee.org/document/10161066), [[Surgical Thread Reconstrcution, ICRA'23]](https://ieeexplore.ieee.org/abstract/document/10161539), [[Tissue Tracking, ICRA'24]](https://arxiv.org/abs/2309.13863)
The integration of 3D scene reconstruction and tracking represents a critical advancement in the realm of autonomous robotic perception, particularly in the context of unstructured environments. Addressing challenges posed by large deformations and the intricacies of texture-less, moistured tissue, and instrument surfaces is imperative for robust and accurate robotic systems. This research delves into the intricacies of scene understanding, path planning, and navigation, emphasizing the pivotal role played by advanced perception techniques. Beyond the specific application in autonomous surgery, the outcomes of this research hold promise for contributing to the broader landscape of robotics research, enhancing adaptability and autonomy across various domains where robotic platforms operate in complex and dynamic environments. As we navigate the intricacies of unstructured scenarios, the integration of cutting-edge 3D scene reconstruction and tracking methodologies stands at the forefront of shaping the future capabilities of autonomous robotic systems.

<table width="920">
	<tr>
		<td width="460">
			<iframe width="460" height="258" src="https://www.youtube.com/embed/tkWgFObuS4w" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen=""></iframe>
		</td>
		<td width="460">
			<iframe width="460" height="258" src="https://www.youtube.com/embed/4Z9up1Pdqxk" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen=""></iframe>
		</td>
	</tr>
</table>