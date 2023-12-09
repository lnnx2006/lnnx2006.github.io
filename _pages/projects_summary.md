---
layout: page
title: Projects
permalink: /projects/
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
			<iframe width="437" height="245" src="https://www.youtube.com/watch?v=hTN-ttMhpcQ" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen=""></iframe>
		</td>
		<td width="438">
			<iframe width="437" height="245" src="https://www.youtube.com/watch?v=ytIRd9gfiQE" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen=""></iframe>
		</td>
	</tr>
  <tr>
		<td width="438">
			<iframe width="437" height="245" src="https://youtu.be/DxUhwZvWw-k" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen=""></iframe>
		</td>
		<td width="438">
			<iframe width="437" height="245" src="https://youtu.be/fVEAxvItFmw" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen=""></iframe>
		</td>
	</tr>
</table>