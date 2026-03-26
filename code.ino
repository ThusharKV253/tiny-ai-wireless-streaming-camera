<!DOCTYPE html>
<html>
<head>
  <title>COCO-SSD Object Detection (Laptop)</title>
  <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@3.3.0"></script>
  <script src="https://cdn.jsdelivr.net/npm/@tensorflow-models/coco-ssd"></script>
  <style>
    #videoElement, #canvas { position: absolute; top: 0; left: 0; }
    #canvas { z-index: 10; }
  </style>
</head>
<body>
  <h2>Live Object Detection with COCO-SSD</h2>
  <video autoplay="true" id="videoElement" width="640" height="480" style="border: 1px solid #ddd"></video>
  <canvas id="canvas" width="640" height="480"></canvas>
  <script>
    const video = document.getElementById('videoElement');
    const canvas = document.getElementById('canvas');
    const ctx = canvas.getContext('2d');
    let model = null;

    async function setupCamera() {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      video.srcObject = stream;
      await new Promise(resolve => video.onloadedmetadata = resolve);
    }

    async function runDetection() {
      model = await cocoSsd.load();
      detectFrame();
    }

    async function detectFrame() {
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      const predictions = await model.detect(video);
      predictions.forEach(pred => {
        ctx.strokeStyle = "#00FFFF";
        ctx.lineWidth = 2;
        ctx.strokeRect(...pred.bbox);
        ctx.font = "16px Arial";
        ctx.fillStyle = "#00FFFF";
        ctx.fillText(pred.class, pred.bbox[0], pred.bbox[1] > 10 ? pred.bbox[1] - 5 : 10);
      });
      requestAnimationFrame(detectFrame);
    }

    setupCamera().then(runDetection);
  </script>
</body>
</html>
