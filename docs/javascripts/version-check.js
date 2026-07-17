(() => {
  const revisionMeta = document.querySelector(
    'meta[name="rolloutlib-docs-revision"]',
  );
  const pageRevision = revisionMeta?.content;
  const placeholder = "__ROLLOUTLIB_DOCS_REVISION__";

  if (!pageRevision || pageRevision === placeholder) {
    return;
  }

  const storageKey = "rolloutlib-docs-revision";
  let previousRevision;

  try {
    previousRevision = window.sessionStorage.getItem(storageKey);
    if (!previousRevision) {
      window.sessionStorage.setItem(storageKey, pageRevision);
      return;
    }
  } catch {
    return;
  }

  if (previousRevision === pageRevision) {
    return;
  }

  const scriptUrl = document.currentScript?.src;
  if (!scriptUrl) {
    return;
  }

  const reconcileRevision = async () => {
    try {
      const revisionUrl = new URL("../revision.json", scriptUrl);
      revisionUrl.searchParams.set("cache_bust", Date.now().toString());

      const response = await fetch(revisionUrl, { cache: "no-store" });
      if (!response.ok) {
        return;
      }

      const payload = await response.json();
      const latestRevision = payload.revision;
      if (typeof latestRevision !== "string" || !latestRevision) {
        return;
      }

      window.sessionStorage.setItem(storageKey, latestRevision);
      if (latestRevision === pageRevision) {
        return;
      }

      const pageUrl = new URL(window.location.href);
      const revisionParameter = latestRevision.slice(0, 12);
      if (pageUrl.searchParams.get("docs_revision") === revisionParameter) {
        return;
      }

      pageUrl.searchParams.set("docs_revision", revisionParameter);
      window.location.replace(pageUrl);
    } catch {
      // Navigation should remain usable if revision discovery is unavailable.
    }
  };

  void reconcileRevision();
})();
