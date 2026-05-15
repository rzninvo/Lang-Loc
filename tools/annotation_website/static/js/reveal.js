// Reveal-on-scroll: cheap IntersectionObserver, no library.
(function () {
  "use strict";
  if (!("IntersectionObserver" in window)) {
    document.querySelectorAll(".reveal").forEach((el) => el.classList.add("is-visible"));
    return;
  }
  const io = new IntersectionObserver((entries, obs) => {
    for (const e of entries) {
      if (e.isIntersecting) {
        e.target.classList.add("is-visible");
        obs.unobserve(e.target);
      }
    }
  }, { rootMargin: "0px 0px -10% 0px", threshold: 0.05 });
  document.querySelectorAll(".reveal").forEach((el) => io.observe(el));
})();
