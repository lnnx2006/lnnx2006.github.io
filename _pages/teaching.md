---
layout: page
permalink: /teaching/
title: teaching
description:
years: [2022, 2015, 2014, 2013, 2012]
nav: true
nav_order: 3
---

I started teaching as a first-year undergraduate student in 2007, at an Euclidean Geometry course at high-school level. During the years that followed and until completing my PhD in 2016, I was a teaching assistant in various courses at _Universidad Nacional de La Plata_, within both the Physics and Mathematics departments (Calculus I & II; Physics I, II & III; General Relativity; Statistical Mechanics; Analytical Mechanics). I thoroughly enjoyed the experience, and consider teaching an integral part of my personal learning experience and academic life.

Apart from university-level courses in Physics and Mathematics, I've taught several mini-courses on competitive programming at various universities throughout Latin America.

<div class="publications">

{%- for y in page.years %}
  <h2 class="year">{{y}}</h2>
  {% bibliography -f teaching -q @*[year={{y}}]* %}
{% endfor %}

</div>