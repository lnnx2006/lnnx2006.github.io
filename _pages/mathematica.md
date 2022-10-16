---
layout: page
permalink: /mathematica/
title: mathematica
description:
nav: false
---

<script>

statementBlock = `If the embedded version below does not work, you can download a copy <a href="PDFURL">here</a>.
<object data="PDFURL#view=FitH&pagemode=none" type="application/pdf" style="width:100%;height:50vh">
  <embed src="PDFURL#view=FitH&pagemode=none" type="application/pdf" />
</object>`

function toggleStatement(elementId, pdfUrl) {
  var codeBlock = document.getElementById(elementId);
  var innerHTML = codeBlock.innerHTML;
  if (innerHTML == "") codeBlock.innerHTML = statementBlock.replaceAll('PDFURL', pdfUrl);
  codeBlock.classList.toggle('open');
}
</script>

I use [Mathematica](https://www.wolfram.com/mathematica/) for symbolic and numerical computations in most of my Physics projects. If you are interested in looking at or reusing the code from any of my [publications](/publications/), let me know by writing to <tt>fidel.s (at) gmail (dot) com</tt>

<hr>

In 2019 I prepared some problems aiming to introduce Mathematica and improve the coding skills of people with a Physics background but not necessarily experienced in functional programming. While the Wolfram language has changed significantly since then, you may still find something useful in the following:

<div id="problemList" class="mathematica_problems">
  <div class="problemEntry">
    <div class="row border-top">
      <div class="title col-sm-8 my-auto">
        Mathematica basics
      </div>
      <div class="col-sm-2">
        <a class="statement btn btn-sm z-depth-0" role="button" onClick="toggleStatement('problem1', '/assets/pdf/1%20-%20Basics.pdf')">STATEMENT</a>
      </div>
      <div class="col-sm-2">
        <a href="/assets/other/1%20-%20Basics.nb" class="btn btn-sm z-depth-0" role="button">SOLUTION</a>
      </div>
    </div>
    <div class="row statement">
      <div id="problem1" class="statement hidden"></div>
    </div>
  </div>
  <div class="problemEntry">
    <div class="row border-top">
      <div class="title col-sm-8 my-auto">
        Ladder operators for the anharmonic oscillator
      </div>
      <div class="col-sm-2">
        <a class="statement btn btn-sm z-depth-0" role="button" onClick="toggleStatement('problem2', '/assets/pdf/2%20-%20Ladder%20Operators.pdf')">STATEMENT</a>
      </div>
      <div class="col-sm-2">
        <a href="/assets/other/2%20-%20Ladder%20Operators.nb" class="btn btn-sm z-depth-0" role="button">SOLUTION</a>
      </div>
    </div>
    <div class="row statement">
      <div id="problem2" class="statement hidden"></div>
    </div>
  </div>
  <div class="problemEntry">
    <div class="row border-top">
      <div class="title col-sm-8 my-auto">
        N-body choreographies
      </div>
      <div class="col-sm-2">
        <a class="statement btn btn-sm z-depth-0" role="button" onClick="toggleStatement('problem3', '/assets/pdf/3%20-%20Choreographies.pdf')">STATEMENT</a>
      </div>
      <div class="col-sm-2">
        <a href="/assets/other/3%20-%20Choreographies.nb" class="btn btn-sm z-depth-0" role="button">SOLUTION</a>
      </div>
    </div>
    <div class="row statement">
      <div id="problem3" class="statement hidden"></div>
    </div>
  </div>
  <div class="problemEntry">
    <div class="row border-top">
      <div class="title col-sm-8 my-auto">
        Finite N Sachdev-Ye-Kitaev model
      </div>
      <div class="col-sm-2">
        <a class="statement btn btn-sm z-depth-0" role="button" onClick="toggleStatement('problem4', '/assets/pdf/4%20-%20Finite%20N%20SYK.pdf')">STATEMENT</a>
      </div>
      <div class="col-sm-2">
        <a href="/assets/other/4%20-%20Finite%20N%20SYK.nb" class="btn btn-sm z-depth-0" role="button">SOLUTION</a>
      </div>
    </div>
    <div class="row statement">
      <div id="problem4" class="statement hidden"></div>
    </div>
  </div>
    <div class="problemEntry">
    <div class="row border-top">
      <div class="title col-sm-8 my-auto">
        Large N Sachdev-Ye-Kitaev model
      </div>
      <div class="col-sm-2">
        <a class="statement btn btn-sm z-depth-0" role="button" onClick="toggleStatement('problem5', '/assets/pdf/5%20-%20Large%20N%20SYK.pdf')">STATEMENT</a>
      </div>
      <div class="col-sm-2">
        <a href="/assets/other/5%20-%20Large%20N%20SYK.nb" class="btn btn-sm z-depth-0" role="button">SOLUTION</a>
      </div>
    </div>
    <div class="row statement">
      <div id="problem5" class="statement hidden"></div>
    </div>
  </div>
  <div class="problemEntry">
    <div class="row border-top">
      <div class="title col-sm-6 my-auto">
        Neural networks in Mathematica
      </div>
      <div class="col-sm-2">
        <a href="/assets/other/6%20-%20Neural%20Networks.zip" class="btn btn-sm z-depth-0" role="button">DATASET</a>
      </div>
      <div class="col-sm-2">
        <a class="statement btn btn-sm z-depth-0" role="button" onClick="toggleStatement('problem6', '/assets/pdf/6%20-%20Neural%20Networks.pdf')">STATEMENT</a>
      </div>
      <div class="col-sm-2">
        <a href="/assets/other/6%20-%20Neural%20Networks.nb" class="btn btn-sm z-depth-0" role="button">SOLUTION</a>
      </div>
    </div>
    <div class="row statement">
      <div id="problem6" class="statement hidden"></div>
    </div>
  </div>
  <div class="problemEntry">
    <div class="row border-top">
      <div class="title col-sm-8 my-auto">
        Tensor models I
      </div>
      <div class="col-sm-2">
        <a class="statement btn btn-sm z-depth-0" role="button" onClick="toggleStatement('problem7', '/assets/pdf/7%20-%20Graph%20Games%20I.pdf')">STATEMENT</a>
      </div>
      <div class="col-sm-2">
        <a href="/assets/other/7%20-%20Graph%20Games%20I.nb" class="btn btn-sm z-depth-0" role="button">SOLUTION</a>
      </div>
    </div>
    <div class="row statement">
      <div id="problem7" class="statement hidden"></div>
    </div>
  </div>
  <div class="problemEntry">
    <div class="row border-top">
      <div class="title col-sm-8 my-auto">
        Tensor models II
      </div>
      <div class="col-sm-2">
        <a class="statement btn btn-sm z-depth-0" role="button" onClick="toggleStatement('problem8', '/assets/pdf/8%20-%20Graph%20Games%20II.pdf')">STATEMENT</a>
      </div>
      <div class="col-sm-2">
        <a href="/assets/other/8%20-%20Graph%20Games%20II.nb" class="btn btn-sm z-depth-0" role="button">SOLUTION</a>
      </div>
    </div>
    <div class="row statement">
      <div id="problem8" class="statement hidden"></div>
    </div>
  </div>
</div>