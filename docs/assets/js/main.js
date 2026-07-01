(() => {
  const videos = document.querySelectorAll('.project-video');

  // Keep the page quiet: start only the video the viewer actively plays.
  videos.forEach((video) => {
    video.addEventListener('play', () => {
      videos.forEach((other) => {
        if (other !== video && !other.paused) other.pause();
      });
    });
  });
})();
