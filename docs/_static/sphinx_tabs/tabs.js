if (!String.prototype.startsWith) {
  Object.defineProperty(String.prototype, 'startsWith', {
    value: function(search, pos) {
      pos = !pos || pos < 0 ? 0 : +pos;
      return this.substring(pos, pos + search.length) === search;
    }
  });
}

// From http://stackoverflow.com/questions/123999/how-to-tell-if-a-dom-element-is-visible-in-the-current-viewport
function elementIsInView (el) {
  if (typeof jQuery === "function" && el instanceof jQuery) {
    el = el[0];
  }

  const  rect = el.getBoundingClientRect();

  return (
    rect.top >= 0 &&
    rect.left >= 0 &&
    rect.bottom <= (window.innerHeight || document.documentElement.clientHeight) &&
    rect.right <= (window.innerWidth || document.documentElement.clientWidth)
  );
}

$(function() {
  // Change container tags <div> -> <a>
  $('.sphinx-menu.menu .item').each(function() {
    var this_ = $(this);
    var a_this = $('<a>');

    a_this.html(this_.html());
    $.each(this_.prop('attributes'), function() {
      a_this.attr(this.name, this.value);
    });

    this_.replaceWith(a_this);
  });

  // We store the data-tab values as sphinx-data-<data-tab value>
  // Add data-tab attribute with the extracted value
  $('.sphinx-menu.menu .item, .sphinx-tab.tab').each(function() {
    var this_ = $(this);
    const prefix = 'sphinx-data-';
    const classes = this_.attr('class').split(/\s+/);
    $.each(classes, function(idx, clazz) {
      if (clazz.startsWith(prefix)) {
        this_.attr('data-tab',
                   clazz.substring(prefix.length));
      }
    });
  });

  // Mimic the Semantic UI behaviour
  $('.sphinx-menu.menu .item').each(function() {
    var this1 = $(this);
    var data_tab = this1.attr('data-tab');

    this1.on('click', function() {
      // Find offset in view
      const offset = (this1.offset().top - $(window).scrollTop());

      // Enable all tabs with this id

      // For each tab group
      $('.sphinx-tabs').each(function() {
        var this2 = $(this);

        // Check if tab group has a tab matching the clicked tab
        var has_tab = false;
        this2.children().eq(0).children().each(function() {
          has_tab |= $(this).attr('data-tab') === data_tab;
        });

        if (has_tab) {
          // Enable just the matching tab
          var toggle = function() {
            var this3 = $(this);
            if (this3.attr('data-tab') === data_tab) {
              this3.addClass('active');
            } else {
              this3.removeClass('active');
            }
          };
          this2.children().eq(0).children('[data-tab]').each(toggle);
          this2.children('[data-tab]').each(toggle);
        }
      });

      // Keep tab with the original view offset
      $(window).scrollTop(this1.offset().top - offset);
    });
  });
});
