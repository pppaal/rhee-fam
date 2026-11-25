// Applies the stored theme and up/down colours before the first paint (spec F.8). Owner: WP3-app.
// A classic (non-module) script loaded in <head>; everything is wrapped in try/catch because storage may throw.
(function () {
  'use strict';
  var root = document.documentElement;
  var read = function (key) {
    try {
      return window.localStorage.getItem(key);
    } catch (e) {
      return null;
    }
  };
  try {
    var theme = read('trendai.theme');
    if (theme === 'light' || theme === 'dark') root.setAttribute('data-theme', theme);
    var updown = read('trendai.updown');
    if (updown !== 'kr' && updown !== 'intl') {
      var match = /[?&]hl=(ko|en)(?:&|$)/.exec(window.location.search);
      var hl = (match && match[1]) || read('trendai.hl');
      if (hl !== 'ko' && hl !== 'en') {
        var langs = (navigator.languages && navigator.languages.length) ? navigator.languages : [navigator.language || ''];
        hl = 'en';
        for (var i = 0; i < langs.length; i += 1) {
          if (String(langs[i]).toLowerCase().indexOf('ko') === 0) { hl = 'ko'; break; }
        }
      }
      updown = hl === 'ko' ? 'kr' : 'intl';
    }
    root.setAttribute('data-updown', updown);
  } catch (e) {
    root.setAttribute('data-updown', 'kr');
  }
}());
