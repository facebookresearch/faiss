try {
  var session = window.sessionStorage || {};
} catch (e) {
  var session = {};
}

window.addEventListener("DOMContentLoaded", () => {
  const allTabs = document.querySelectorAll('.sphinx-tabs-tab');
  const tabLists = document.querySelectorAll('[role="tablist"]');

  allTabs.forEach(tab => {
    tab.addEventListener("click", changeTabs);
  });

  tabLists.forEach(tabList => {
    tabList.addEventListener("keydown", keyTabs);
  });

  const lastSelected = session.getItem('sphinx-tabs-last-selected');
  if (lastSelected != null) selectGroupedTabs(lastSelected);
});

/**
 * Key focus left and right between sibling elements using arrows
 * @param  {Node} e the element in focus when key was pressed
 */
function keyTabs(e) {
    const tab = e.target;
    let nextTab = null;
    if (e.keyCode === 39 || e.keyCode === 37) {
      tab.setAttribute("tabindex", -1);
      // Move right
      if (e.keyCode === 39) {
        nextTab = tab.nextElementSibling;
        if (nextTab === null) {
          nextTab = tab.parentNode.firstElementChild;
        }
      // Move left
      } else if (e.keyCode === 37) {
        nextTab = tab.previousElementSibling;
        if (nextTab === null) {
          nextTab = tab.parentNode.lastElementChild;
        }
      }
    }

    if (nextTab !== null) {
      nextTab.setAttribute("tabindex", 0);
      nextTab.focus();
    }
}

/**
 * Select or deselect clicked tab. If a group tab
 * is selected, also select tab in other tabLists.
 * @param  {Node} e the element that was clicked
 */
function changeTabs(e) {
  const target = e.target;
  const selected = target.getAttribute("aria-selected") === "true";
  const positionBefore = target.parentNode.getBoundingClientRect().top;

  deselectTabset(target);

  if (!selected) {
    selectTab(target);
    const name = target.getAttribute("name");
    selectGroupedTabs(name, target.id);

    if (target.classList.contains("group-tab")) {
      // Persist during session
      session.setItem('sphinx-tabs-last-selected', name);
    }
  }

  const positionAfter = target.parentNode.getBoundingClientRect().top;
  const positionDelta = positionAfter - positionBefore;
  // Scroll to offset content resizing
  window.scrollTo(0, window.scrollY + positionDelta);
}

function selectTab(target) {
  target.setAttribute("aria-selected", true);

  // Show the associated panel
  document
    .getElementById(target.getAttribute("aria-controls"))
    .removeAttribute("hidden");
}

function selectGroupedTabs(name, clickedId=null) {
  const groupedTabs = document.querySelectorAll(`.sphinx-tabs-tab[name="${name}"]`);
  const tabLists = Array.from(groupedTabs).map(tab => tab.parentNode);

  tabLists
    .forEach(tabList => {
      // Don't want to change the tabList containing the clicked tab
      const clickedTab = tabList.querySelector(`[id="${clickedId}"]`);
      if (clickedTab === null ) {
        // Select first tab with matching name
        const tab = tabList.querySelector(`.sphinx-tabs-tab[name="${name}"]`);
        deselectTabset(tab);
        selectTab(tab);
      }
    })
}

function deselectTabset(target) {
  const parent = target.parentNode;
  const grandparent = parent.parentNode;

  // Hide all tabs in current tablist, but not nested
  Array.from(parent.children)
  .forEach(t => t.setAttribute("aria-selected", false));

  // Hide all associated panels
  Array.from(grandparent.children)
    .slice(1)  // Skip tablist
    .forEach(p => p.setAttribute("hidden", true));
}
