$.fn.isInViewport = function () {
  var elementTop = $(this).offset().top;
  var elementBottom = elementTop + $(this).outerHeight();
  var viewportTop = $(window).scrollTop();
  var viewportBottom = viewportTop + $(window).height();
  return elementBottom > viewportTop && elementTop < viewportBottom;
};

var allPublications = null;

function disableLink(linkId) {
  const link = document.getElementById(linkId);
  link.style.pointerEvents = 'none';
  link.classList.add('active');
}

function enableLink(linkId) {
  const link = document.getElementById(linkId);
  link.style.pointerEvents = 'auto';    // Restores clickability
  link.classList.remove('active');
}

function playAllVideos() {
  $("#pub-container video").each(function () {
    this.play();
  });
}

function publicationBySelected() {
  var a = $("#publication-by-selected")
  if (a.hasClass("activated")) {
    return;
  }

  $("#pub-container .subtitle a").removeClass("activated");
  $("#pub-container .subtitle-aux a").removeClass("activated");
  a.addClass("activated");

  $("#pub-card-container").html("");
  for (var pubId = 0; pubId < allPublications.length; pubId++) {
    var pub = $(allPublications[pubId]);
    if (pub.data("selected") == true) {
      $("#pub-card-container").append(pub);
    }
  }
  playAllVideos();
}

function publicationByDate() {
  var a = $("#publication-by-date")
  if (a.hasClass("activated")) {
    return;
  }

  $("#pub-container .subtitle a").removeClass("activated");
  $("#pub-container .subtitle-aux a").removeClass("activated");
  a.addClass("activated");

  $("#pub-card-container").html("");
  for (var pubId = 0; pubId < allPublications.length; pubId++) {
    if (pubId == 0 || $(allPublications[pubId - 1]).data("year") != $(allPublications[pubId]).data("year")) {
      var year = $(allPublications[pubId]).data("year");
      $("#pub-card-container").append($("<h5 id='year-" + year.toString() + "' style='margin-top: 40px;'>" + year.toString() + "</h5>"));
    }
    $("#pub-card-container").append(allPublications[pubId]);
  }
  playAllVideos();

  // disableLink('showAllByTopicPubLink');
  // enableLink('showAllByDatePubLink');

  return true;
}

function publicationByTopic() {
  publicationByTopicInner();
  publicationByTopicSpecificInner($("#pub-container .subtitle-aux a:first"));
  playAllVideos();

  // disableLink('showAllByDatePubLink');
  // enableLink('showAllByTopicPubLink');

  return true;
}

function publicationByTopicInner() {
  var a = $("#publication-by-topic")
  if (a.hasClass("activated")) {
    return;
  }
  $("#pub-container .subtitle a").removeClass("activated");
  a.addClass("activated");

  $("#pub-card-container").html("");
  for (var topicId in allTopics) {
    var topic = allTopics[topicId].name;
    var topicTitle = allTopics[topicId].title;
    // var topicTitle = topic.split("-").map(function (a) { return a[0].toUpperCase() + a.substr(1).toLowerCase(); }).join(" ");
    // $("#pub-card-container").append($("<h5 id='topic-" + topic + "'>" + topicTitle + "</h5>"));
    $("#pub-card-container").append($("<h5 id='topic-" + topic + "' style='margin-top: 40px;'>" + topicTitle + "</h5>"));
    for (var pubId = 0; pubId < allPublications.length; pubId++) {
      var pub = $(allPublications[pubId]);
      if (pub.data("topic").indexOf(topic) != -1) {
        $("#pub-card-container").append(pub);
      }
    }
  }
}

function publicationByTopicSpecificInner(a) {
  if ($(a).hasClass("activated")) {
    return false;
  }

  $("#pub-container .subtitle-aux a").removeClass("activated");
  $(a).addClass("activated");
}

function publicationByTopicSpecific(a) {
  publicationByTopicInner();
  publicationByTopicSpecificInner(a);

  var hash = a.hash;
  $(hash).prop('id', hash.substr(1) + '-noscroll');
  window.location.hash = hash;
  $(hash + '-noscroll').prop('id', hash.substr(1));

  if (!$(hash).isInViewport()) {
    $('html, body').animate({
      scrollTop: $(hash).offset().top
    }, 1000, function () {
    });
  }

  playAllVideos();
  return false;
}

function toggleNews(linkElement) {
  $("#news-more").slideToggle();
  var e = $("#news-more-button");
  if (e.data("expanded")) {
    e.html("Click to hide old news ...");
    e.data("expanded", false);
  } else {
    e.html("Click to show older news ...");
    e.data("expanded", true);
  }
}

$(function () {
  getRealSize = function (bgImg) {
    var img = new Image();
    img.src = bgImg.attr("src");
    var width = img.width,
      height = img.height;
    return {
      width: width,
      height: height
    }
  };

  getRealWindowSize = function () {
    var winWidth = null,
      winHeight = null;
    if (window.innerWidth) winWidth = window.innerWidth;
    else if ((document.body) && (document.body.clientWidth)) winWidth = document.body.clientWidth;
    if (window.innerHeight) winHeight = window.innerHeight;
    else if ((document.body) && (document.body.clientHeight)) winHeight = document.body.clientHeight;
    if (document.documentElement && document.documentElement.clientHeight && document.documentElement.clientWidth) {
      winHeight = document.documentElement.clientHeight;
      winWidth = document.documentElement.clientWidth
    }
    return {
      width: winWidth,
      height: winHeight
    }
  };

  fullBg = function () {
    var bgImg = $("#background");
    var mainContainer = $("#main");
    var firstFire = null;

    if (bgImg.length == 0) {
      return;
    }

    function resizeImg() {
      var realSize = getRealSize(bgImg);
      var imgWidth = realSize.width;
      var imgHeight = realSize.height;

      if (imgWidth == 0 || imgHeight == 0) {
        setTimeout(function () {
          resizeImg();
        }, 200);
      }

      console.log(realSize);
      var realWinSize = getRealWindowSize();
      var winWidth = realWinSize.width;
      var winHeight = realWinSize.height;
      var widthRatio = winWidth / imgWidth;
      var heightRatio = winHeight / imgHeight;
      console.log(realWinSize);
      if (widthRatio > heightRatio) {
        bgImg.width(imgWidth * widthRatio + 'px').height(imgHeight * widthRatio + 'px').css({
          'top':
            -(imgHeight * widthRatio - winHeight) / 10 * 5 + 'px', 'left': '0'
        })
      } else {
        bgImg.width(imgWidth * heightRatio + 'px').height(imgHeight * heightRatio + 'px').css({
          'left':
            -(imgWidth * heightRatio - winWidth) / 10 * 3 + 'px', 'top': '0'
        })
      }
      // mainContainer.css({
      //     width: winWidth,
      //     height: winHeight
      // });
    }

    resizeImg();
    window.onresize = function () {
      if (firstFire === null) {
        firstFire = setTimeout(function () {
          resizeImg();
          firstFire = null
        }, 100)
      }
    }
  };

  targetColor = $("#main-content-container .name").css("color");
  animatedLink = function (speed) {
    $("#main-content-container .col-link li").hover(function () {
      $(this).find('.icon').animate({
        color: targetColor,
        borderColor: targetColor
      }, speed);
      $(this).find('.caption').animate({
        color: targetColor
      })
    }, function () {
      $(this).find('.icon').animate({
        borderColor: '#cccccc',
        color: '#cccccc'
      }, speed);
      $(this).find('.caption').animate({
        color: '#cccccc'
      })
    })
  };

  // fullBg();
  // animatedLink(400);

  allPublications = $("#pub-card-container .pub-card");
  allTopicsLink = $("#pub-container .subtitle-aux a");
  allTopics = [];
  for (var topicId = 0; topicId < allTopicsLink.length; topicId++) {
    allTopics.push({ name: $(allTopicsLink[topicId]).data("topic"), title: $(allTopicsLink[topicId]).html() });
  }

  // $("#publication-by-selected").click();
  $("#publication-by-date").click();
  $("#pub-card-container").removeClass("hide");
});