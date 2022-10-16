---
layout: page
permalink: /competitive_programming/
title: competitive programming
description:
nav: false
---

<script>
function filterNames() {
  var textFilter = document.getElementById("textFilter");
  var filter = textFilter.value.toUpperCase();
  var problemList = document.getElementById("problemList");
  rows = problemList.getElementsByClassName("problemEntry");

  // Loop through all table rows, and hide those which don't match the search query
  for (i = 0; i < rows.length; i++) {
    problemName = rows[i].getElementsByClassName("problemData")[0];
    if (problemName) {
      txtValue = problemName.textContent || problemName.innerText;
      if (txtValue.toUpperCase().indexOf(filter) > -1) {
        rows[i].style.display = "";
      } else {
        rows[i].style.display = "none";
      }
    }
  }
}

var preCode = '<pre class="code"><code class="language-c++">';
var postCode = '</code></pre>';

String.prototype.toHtmlEntities = function() {
    return this.replace(/./gm, function(s) {
        // return "&#" + s.charCodeAt(0) + ";";
        return (s.match(/[a-z0-9\s]+/i)) ? s : "&#" + s.charCodeAt(0) + ";";
    });
};

function toggleCode(elementId, code_path) {
  var codeBlock = document.getElementById(elementId);
  var innerHTML = codeBlock.innerHTML;
  if (innerHTML == "") {
	fetch('https://api.github.com/repos/fidel-schaposnik/icpc-solutions/contents/'+code_path)
		.then(response => response.json())
		.then( data => codeBlock.innerHTML = preCode+atob(data['content']).toHtmlEntities()+postCode);
  }
  codeBlock.toggleClass('open');
}
</script>

I started participating in the _ACM International Collegiate Programming Contest_ ([ICPC](https://icpc.global/)) in 2006, as a member of team "La JiRaFa" along with [Joaquin Rodrigues Jacinto](https://sites.google.com/site/joaquinrj/home) and [Ramiro Lafuente](https://sites.google.com/view/ramlaf/home). We qualified to the ACM--ICPC World Finals twice, in 2009 (Stockholm) and 2010 (Harbin). I've also ocassionally participated in [CodeForces](https://codeforces.com/profile/fidels), [TopCoder](https://www.topcoder.com/members/fidels), CodeJam, HackerCup, and other individual programming competitions.

After retiring as a contestant, I moved on to problem-setting and jury duties both at the [Argentinian](http://torneoprogramacion.com.ar/) and Latin-American regional level of ACM--ICPC. I've also taught intensive mini-courses on competitive programming at various Latin-American universities. You can find more details, as well as some of my lectures on various topics in competitive programming, in my [teaching](/teaching/) page.

I am (slowly) adding some of the problems and solutions I used to practice, as well as problems I wrote myself, to [this repository](https://github.com/fidel-schaposnik/icpc-solutions). The following is a list of all {{ site.data.problem_db.total_problems}} problems currently in the database, with links to their statements and solutions. More problems, solutions and tags are added as time permits.

<input type="text" id="textFilter" onkeyup="filterNames()" placeholder="Search for problem names, tags..." style="width:100%;padding:10px;color:var(--global-text-color);background-color:var(--global-bg-color);">

<div id="problemList" class="problems">
  {%- for problem in site.data.problem_db.problems -%}
  <div class="problemEntry">
    <div class="row border-top">
	  <div class="col-sm-auto my-auto">
		{%- if problem.judge == "LiveArchive"-%}
		<img src="/assets/img/LiveArchive.png" alt="LiveArchive">
		{%- endif -%}
	  </div>
	  <div class="col-sm-8 my-auto">
	    <div class="row problemData">
		  <div class="col-sm-auto">
	        <a href="https://icpcarchive.ecs.baylor.edu/index.php?option=com_onlinejudge&page=show_problem&problem={{ problem.problem_id }}">{{ problem.problem_name }}</a>
		  </div>
		  {%- for tag in problem.tags -%}
		  <div class="col-sm-auto"><abbr class="badge badge-light">{{ tag }}</abbr></div>
		  {%- endfor -%}
		</div>
	  </div>
	  <div class="col-sm-1"><a href="https://github.com/fidel-schaposnik/icpc-solutions/raw/master/{{ problem.directory | uri_escape}}/{{ problem.statement | uri_escape}}" class="btn btn-sm z-depth-0" role="button">PDF</a></div>
	  <div class="col-sm-1"><a class="code btn btn-sm z-depth-0" role="button" onClick="toggleCode({{ problem.problem_id }}, '{{ problem.directory | uri_escape}}/{{ problem.code | uri_escape}}')">CODE</a></div>
    </div>
	<div class="row code">
	  <div id="{{ problem.problem_id }}" class="code hidden"></div>
	</div>
  </div>
  {%- endfor -%}
</div>