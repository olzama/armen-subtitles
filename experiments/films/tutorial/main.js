let current = 0;

function buildDots() {
  ["slide-dots", "slide-dots-top"].forEach(id => {
    const container = document.getElementById(id);
    container.innerHTML = "";
    slides.forEach((slide, i) => {
      const btn = document.createElement("button");
      btn.className = "dot" + (i === current ? " active" : "");
      btn.textContent = i + 1;
      btn.title = slide.title;
      btn.onclick = () => { current = i; render(); window.scrollTo(0, 0); };
      container.appendChild(btn);
    });
  });
}

function render() {
  const slide = slides[current];
  document.getElementById("step-indicator").textContent =
    `Step ${current + 1} of ${slides.length}`;
  document.getElementById("slide-title").textContent = slide.title;
  document.getElementById("slide-content").innerHTML = slide.content();
  document.getElementById("btn-prev").disabled = current === 0;
  document.getElementById("btn-next").disabled = current === slides.length - 1;
  buildDots();
}

function navigate(dir) {
  const next = current + dir;
  if (next >= 0 && next < slides.length) {
    current = next;
    render();
    window.scrollTo(0, 0);
  }
}

document.addEventListener("keydown", e => {
  if (e.key === "ArrowRight" || e.key === "ArrowDown") navigate(1);
  if (e.key === "ArrowLeft"  || e.key === "ArrowUp")   navigate(-1);
});

render();
