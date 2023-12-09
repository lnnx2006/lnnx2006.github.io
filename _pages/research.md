---
layout: page
title: Research
permalink: /research/
description: The summary of my projects.
nav: true
nav_order: 1
display_categories: 
horizontal: false
---

## Physics-based Modeling and Differentiable Simulation
The focus of this work is on the **modeling and simulation of robotic scenes in a uniformed representation**. Real-time robotic applications encompass a diverse range of materials and tools, including soft, deformable, fluid objects, and rigid, articulated, or soft tools. The challenge of addressing this diversity is met by leveraging computer graphics methods to manage different material properties, kinematics, and dynamics. The chosen approach employs the position-based dynamics (PBD) method to efficiently simulate and model various objects in real-time, incorporating particle dynamics with physical and geometric constraints. This formulation offers the advantages of rapid positional adjustment and stable convergence in parallel. Simultaneously, all constraints can be defined in a differentiable manner, establishing a strong connection to data-driven methods (e.g., neural network).

The unified simulation and modeling approach has been applied, particularly in surgical contexts. This application extends to surgical tools such as da Vinci Research Kit robotic tools, deformable volumes, soft tissue, and viscous fluids like blood. In the context of surgical navigation and manipulation tasks, particle-based tool-tissue interaction modeling with 2D mesh structures has been considered. For general robotics applications, various structural objects with different types of geometrical constraints have been modeled to capture physical effects. For instance, in rope manipulation, universal particle-based modeling with serial-linked robotic manipulators has been employed. Safety constraints have also been taken into account for cloth trajectory optimization. Additionally, geometrical models like Bézier curves for soft robots and polynomial splines for surgical threads have been used, both of which can be easily formalized into curve-based constraints with discrete sets of particles representing thin structural objects. In summary, the proposed physics-based modeling approach can represent a broad spectrum of multi-stiffness objects and tools for dexterous manipulation, as well as robot modeling from rigid to soft structures.

<table width="876">
	<tr>
		<td width="438">
			<iframe width="437" height="245" src="https://www.youtube.com/embed/hTN-ttMhpcQ" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen=""></iframe>
		</td>
		<td width="438">
			<iframe width="437" height="245" src="https://www.youtube.com/embed/DxUhwZvWw-k" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen=""></iframe>
		</td>
	</tr>
  <tr>
		<td width="438">
			<iframe width="437" height="245" src="https://www.youtube.com/embed/fVEAxvItFmw" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen=""></iframe>
		</td>
		<td width="438">
			<iframe width="437" height="245" src="https://www.youtube.com/embed/X_UPhL_TjTI" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen=""></iframe>
		</td>
	</tr>
</table>

<br/><br/>

## Advanced Planning, Control and Learning
The characterization of **advanced planning, control and learning modules for integration into the robotic system** is another crucial area of my past research. In particular, my recently developed positional constraints-based multi-body dynamics (PBD) framework can solve state-of-the-art control problems. More importantly, it supports arbitrary differentiation for an objective function with respect to all physical variables involving control input, system states, initial conditions and parameters etc. Therefore, many robotic inverse problems (such as parameter identification, motion control and planning, and trajectory optimization) can be formulated into constrained minimization solutions where the state variables are subject to equality and inequality constraints derived from physical principles. 

Thanks to the differentiable property with ready-to-use gradients using our PBD modeling approach, several downstream robotic applications are carried out, including impedance control and trajectory planning of compliant serial elastic Baxter arms, shape control of linear-deformable objects, trajectory optimization of cloth, and blood fluid with trajectory planning and model-predictive control (MPC). More significantly, our framework can be transferred to *real-to-sim/sim-to-real* experimental setups, closing the "gap" between the simulated systems from real-world cases. I've made an attempt to build an online, continuous, *real-to-sim* registration method for 3D visual perception (surface point cloud) and physics-based PBD simulation (volumetric mesh). The application is soft tissue manipulation using da Vinci surgical tools. To summarize, my work has a high potential for efficiently solving model-based or data-driven control/planning problems with a differentiable optimization formulation. Additionally, many control policies can be trained in an end-to-end fashion using learning approaches, such as our work on 4D lung motion prediction, scene reconstruction.

<table width="876">
	<tr>
		<td width="438">
			<iframe width="437" height="245" src="https://www.youtube.com/embed/RTl-egsjKvM" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen=""></iframe>
		</td>
		<td width="438">
			<iframe width="437" height="245" src="https://www.youtube.com/embed/ytIRd9gfiQE" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen=""></iframe>
		</td>
	</tr>
</table>