
let currentTopic = 'all';
let showSelectedMode = true;

// Make "Show Selected" the default once the page loads
window.onload = () => {
  showTopNews();
  showAllPapers();
  // showSelectedPapers();
  // showTopTalks();
};

function disableLink(linkId) {
  const link = document.getElementById(linkId);
  if (!link) return;
  link.classList.add('active');
}

function enableLink(linkId) {
  const link = document.getElementById(linkId);
  if (!link) return;
  link.style.pointerEvents = 'auto';
  link.classList.remove('active');
}

function showTopNews() {
  const items = document.querySelectorAll('#newsList li');
  items.forEach((item, idx) => {
    item.style.display = idx < 6 ? 'list-item' : 'none';
  });
  disableLink('newsTopLink');
  enableLink('newsAllLink');
}

function showAllNews() {
  const items = document.querySelectorAll('#newsList li');
  items.forEach(item => item.style.display = 'list-item');
  disableLink('newsAllLink');
  enableLink('newsTopLink');
}

function showSelectedPapers() {
  showSelectedMode = true;
  applyFilters();

  // Disable and underline "Show Selected" link
  disableLink('showSelectedLink');
  // Enable "Show All" link
  enableLink('showAllLink');
}

function showAllPapers() {
  showSelectedMode = false;
  currentTopic = 'all';
  document.querySelectorAll('.topic-btn').forEach(btn => btn.classList.remove('active'));
  const allTopicsButton = document.getElementById('topic-all');
  if (allTopicsButton) allTopicsButton.classList.add('active');
  applyFilters();

  // Disable and underline "Show All" link
  disableLink('showAllLink');
  // Keep this action available so it can reset an active topic filter.
  enableLink('showSelectedLink');
}

function filterByTopic(topic) {
  currentTopic = topic;
  // toggle button active state
  document.querySelectorAll('.topic-btn').forEach(btn => btn.classList.remove('active'));
  const btn = document.getElementById(`topic-${topic}`) || document.getElementById('topic-all');
  if (btn) btn.classList.add('active');

  if (topic === 'all') {
    disableLink('showAllLink');
  } else {
    enableLink('showAllLink');
  }

  applyFilters();
}

function applyFilters() {
  const allPapers = document.querySelectorAll('.paperrow');
  allPapers.forEach(paper => {
    const topics = (paper.getAttribute('data-topics') || '').split(',').map(t => t.trim()).filter(Boolean);
    const topicMatches = currentTopic === 'all' || topics.includes(currentTopic);
    const visible = topicMatches;
    paper.style.display = visible ? '' : 'none';
  });
  // allPapers.forEach(paper => {
  //   const isSelected = paper.getAttribute('data-selected') === 'true';
  //   const topics = (paper.getAttribute('data-topics') || '').split(',').map(t => t.trim()).filter(Boolean);
  //   const topicMatches = currentTopic === 'all' || topics.includes(currentTopic);
  //   const visible = (showSelectedMode ? isSelected : true) && topicMatches;
  //   paper.style.display = visible ? 'list-item' : 'none';
  // });
}
