---
layout: page
permalink: /publications/
title: publications
description:
years: [2022, 2021, 2020, 2019]
nav: true
nav_order: 1
---
<!-- _pages/publications.md -->

<script>
function filterSubject(filter) {
  var list = document.getElementById("publicationList");
  var rows = list.getElementsByClassName("row");
  
  // Loop through all rows, hide those which don't match the selected filter
  for (i = 0; i < rows.length; i++) {
    var abbr = rows[i].getElementsByClassName("abbr")[0];
    if (abbr) {
      var txtValue = abbr.textContent || abbr.innerText;
      if (txtValue.indexOf(filter) > -1) {
        rows[i].style.display = "";
      } else {
        rows[i].style.display = "none";
      }
    }
  }
  
  // Loop through all sections, hide those which are empty
  var years = list.getElementsByClassName("year");
  for (i = 0; i < years.length; i++) {
    var count = 0;
    for (j = 0; j < rows.length; j++) {
	  var section_tag = rows[j].getElementsByClassName("section-tag")[0];
	  if (section_tag.textContent == years[i].textContent && rows[j].style.display == "") { count++; }
	}
	if (count != 0) {
	  years[i].style.display = "";
	} else {
	  years[i].style.display = "none";
	}
  }
}
</script>

This is a list of my publications in reverse-chronological order. You can use the buttons below to filter according to the keywords of research areas.

<center>
<abbr class="{{site.data.badge_colors['darkgrey']}}" onclick="filterSubject('')" style="cursor: pointer;">all</abbr>&ensp;
<abbr class="{{site.data.badge_colors['yellow']}}" onclick="filterSubject('surgical_autonomy')" style="cursor: pointer;">surgical_autonomy</abbr>&ensp;
<!-- <abbr class="{{site.data.badge_colors['cyan']}}" onclick="filterSubject('physics.bio-ph')" style="cursor: pointer;">physics.bio-ph</abbr>&ensp;
<abbr class="{{site.data.badge_colors['green']}}" onclick="filterSubject('cond-mat.stat-mech')" style="cursor: pointer;">cond-mat.stat-mech</abbr> -->
</center> 

<div id="publicationList" class="publications">

{%- for y in page.years %}
  <h2 class="year">{{y}}</h2>
  {% bibliography -f publications_fei -q @*[year={{y}}]* %}
{% endfor %}

</div>
