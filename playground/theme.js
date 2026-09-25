// Vahini playground: follow the vahinitech.com theme when it is served.
// theme/vahini-theme.css (supplied by the deployment, see README.md) sets
// --vahini-* tokens. The site is light-only, so with a site theme the page
// stays light instead of switching to its own dark palette. Runs in <head>,
// after the stylesheets, so there is no flash of the dark colours.
"use strict";
(function () {
  const css = getComputedStyle(document.documentElement);
  if (css.getPropertyValue("--vahini-theme").trim() &&
      css.getPropertyValue("--vahini-color-scheme").trim() === "light") {
    document.documentElement.setAttribute("data-theme", "light");
  }
})();
