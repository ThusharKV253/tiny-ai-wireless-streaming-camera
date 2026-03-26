/* ESP32-CAM Edge Impulse with WebSocket for Real-time Updates
 * Optimized for performance - No lag!
 * Arduino IDE Code
 */

#include <miniproject_inferencing.h>
#include "edge-impulse-sdk/dsp/image/image.hpp"
#include "esp_camera.h"
#include "esp_http_server.h"
#include <WiFi.h>
#include <ESPmDNS.h>
#include <WebSocketsServer.h>
#include <ArduinoJson.h>

/* WiFi Configuration ------------------------------------------------------ */
const char* ssid = "Thushar";
const char* password = "12345678";
const char* hostname = "esp32cam";

/* CONFIDENCE THRESHOLD SETTINGS ------------------------------------------- */
#define CONFIDENCE_THRESHOLD 0.65
#define MIN_DETECTION_CONFIDENCE 0.50

/* Camera Model Selection -------------------------------------------------- */
#define CAMERA_MODEL_AI_THINKER

#if defined(CAMERA_MODEL_AI_THINKER)
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27
#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22
#endif

/* Constants --------------------------------------------------------------- */
#define EI_CAMERA_RAW_FRAME_BUFFER_COLS  320
#define EI_CAMERA_RAW_FRAME_BUFFER_ROWS  240
#define EI_CAMERA_FRAME_BYTE_SIZE        3
#define PART_BOUNDARY "123456789000000000000987654321"

/* Global Variables -------------------------------------------------------- */
static bool is_initialised = false;
uint8_t *snapshot_buf;

httpd_handle_t camera_httpd = NULL;
WebSocketsServer webSocket = WebSocketsServer(81);  // WebSocket on port 81

static String ip_address = "";
static unsigned long inference_count = 0;
static unsigned long valid_detection_count = 0;

static const char* _STREAM_CONTENT_TYPE = "multipart/x-mixed-replace;boundary=" PART_BOUNDARY;
static const char* _STREAM_BOUNDARY = "\r\n--" PART_BOUNDARY "\r\n";
static const char* _STREAM_PART = "Content-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n";

/* Camera Configuration ---------------------------------------------------- */
static camera_config_t camera_config = {
    .pin_pwdn = PWDN_GPIO_NUM,
    .pin_reset = RESET_GPIO_NUM,
    .pin_xclk = XCLK_GPIO_NUM,
    .pin_sscb_sda = SIOD_GPIO_NUM,
    .pin_sscb_scl = SIOC_GPIO_NUM,
    .pin_d7 = Y9_GPIO_NUM,
    .pin_d6 = Y8_GPIO_NUM,
    .pin_d5 = Y7_GPIO_NUM,
    .pin_d4 = Y6_GPIO_NUM,
    .pin_d3 = Y5_GPIO_NUM,
    .pin_d2 = Y4_GPIO_NUM,
    .pin_d1 = Y3_GPIO_NUM,
    .pin_d0 = Y2_GPIO_NUM,
    .pin_vsync = VSYNC_GPIO_NUM,
    .pin_href = HREF_GPIO_NUM,
    .pin_pclk = PCLK_GPIO_NUM,
    .xclk_freq_hz = 20000000,
    .ledc_timer = LEDC_TIMER_0,
    .ledc_channel = LEDC_CHANNEL_0,
    .pixel_format = PIXFORMAT_JPEG,
    .frame_size = FRAMESIZE_QVGA,
    .jpeg_quality = 12,
    .fb_count = 2,
    .fb_location = CAMERA_FB_IN_PSRAM,
    .grab_mode = CAMERA_GRAB_WHEN_EMPTY,
};

/* Function Declarations --------------------------------------------------- */
bool ei_camera_init(void);
bool ei_camera_capture(uint32_t img_width, uint32_t img_height, uint8_t *out_buf);
static int ei_camera_get_data(size_t offset, size_t length, float *out_ptr);
void webSocketEvent(uint8_t num, WStype_t type, uint8_t * payload, size_t length);
void sendInferenceResult(String status, String animal, float confidence, int processingTime);

/* Simple HTML (loads external webpage) ------------------------------------ */
static const char PROGMEM INDEX_HTML[] = R"rawliteral(
<!DOCTYPE html>
<html>
<head>
    <title>ESP32-CAM Animal Detection</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <meta charset="UTF-8">
</head>
<body>
    <h1>Loading...</h1>
    <p>Opening external interface...</p>
    <script>
        // Get IP from current URL and redirect to external page
        const currentIP = window.location.hostname;
        window.location.href = http://192.168.137.7:8000/index.html?esp32ip=${currentIP};
    </script>
</body>
</html>
)rawliteral";

/* HTTP Handler - Index ---------------------------------------------------- */
static esp_err_t index_handler(httpd_req_t *req) {
    httpd_resp_set_type(req, "text/html");
    return httpd_resp_send(req, INDEX_HTML, strlen(INDEX_HTML));
}

/* HTTP Handler - MJPEG Stream --------------------------------------------- */
static esp_err_t stream_handler(httpd_req_t *req) {
    camera_fb_t * fb = NULL;
    esp_err_t res = ESP_OK;
    size_t _jpg_buf_len = 0;
    uint8_t * _jpg_buf = NULL;
    char part_buf[64];

    res = httpd_resp_set_type(req, _STREAM_CONTENT_TYPE);
    if(res != ESP_OK) return res;

    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");

    while(true) {
        fb = esp_camera_fb_get();
        if (!fb) {
            Serial.println("Camera capture failed");
            res = ESP_FAIL;
            break;
        }

        if(fb->format != PIXFORMAT_JPEG) {
            bool jpeg_converted = frame2jpg(fb, 80, &_jpg_buf, &_jpg_buf_len);
            esp_camera_fb_return(fb);
            fb = NULL;
            if(!jpeg_converted) {
                Serial.println("JPEG compression failed");
                res = ESP_FAIL;
                break;
            }
        } else {
            _jpg_buf_len = fb->len;
            _jpg_buf = fb->buf;
        }

        if(res == ESP_OK) {
            size_t hlen = snprintf(part_buf, 64, _STREAM_PART, _jpg_buf_len);
            res = httpd_resp_send_chunk(req, part_buf, hlen);
        }
        if(res == ESP_OK) {
            res = httpd_resp_send_chunk(req, (const char *)_jpg_buf, _jpg_buf_len);
        }
        if(res == ESP_OK) {
            res = httpd_resp_send_chunk(req, _STREAM_BOUNDARY, strlen(_STREAM_BOUNDARY));
        }
        
        if(fb) {
            esp_camera_fb_return(fb);
            fb = NULL;
            _jpg_buf = NULL;
        } else if(_jpg_buf) {
            free(_jpg_buf);
            _jpg_buf = NULL;
        }
        
        if(res != ESP_OK) break;
    }
    
    return res;
}

/* Start HTTP Server ------------------------------------------------------- */
void startCameraServer() {
    httpd_config_t config = HTTPD_DEFAULT_CONFIG();
    config.server_port = 80;
    config.ctrl_port = 32768;

    httpd_uri_t index_uri = {
        .uri = "/",
        .method = HTTP_GET,
        .handler = index_handler,
        .user_ctx = NULL
    };

    httpd_uri_t stream_uri = {
        .uri = "/stream",
        .method = HTTP_GET,
        .handler = stream_handler,
        .user_ctx = NULL
    };

    Serial.println("Starting HTTP server...");
    if (httpd_start(&camera_httpd, &config) == ESP_OK) {
        httpd_register_uri_handler(camera_httpd, &index_uri);
        httpd_register_uri_handler(camera_httpd, &stream_uri);
        Serial.println("✅ HTTP server started");
    } else {
        Serial.println("❌ HTTP server failed");
    }
}

/* WebSocket Event Handler ------------------------------------------------- */
void webSocketEvent(uint8_t num, WStype_t type, uint8_t * payload, size_t length) {
    switch(type) {
        case WStype_DISCONNECTED:
            Serial.printf("[%u] Disconnected!\n", num);
            break;
        case WStype_CONNECTED:
            {
                IPAddress ip = webSocket.remoteIP(num);
                Serial.printf("[%u] Connected from %d.%d.%d.%d\n", num, ip[0], ip[1], ip[2], ip[3]);
                
                // Send connection confirmation
                StaticJsonDocument<200> doc;
                doc["type"] = "connection";
                doc["status"] = "connected";
                doc["ip"] = ip_address;
                String json;
                serializeJson(doc, json);
                webSocket.sendTXT(num, json);
            }
            break;
    }
}

/* Send Inference Result via WebSocket ------------------------------------- */
void sendInferenceResult(String status, String animal, float confidence, int processingTime) {
    StaticJsonDocument<512> doc;
    
    doc["type"] = "inference";
    doc["status"] = status;
    doc["animal"] = animal;
    doc["confidence"] = confidence;
    doc["processing_time"] = processingTime;
    doc["inference_count"] = inference_count;
    doc["valid_detections"] = valid_detection_count;
    doc["timestamp"] = millis();
    
    String json;
    serializeJson(doc, json);
    
    webSocket.broadcastTXT(json);
}

/* WiFi Connection --------------------------------------------------------- */
bool connectToWiFi() {
    Serial.println("\n===========================================");
    Serial.printf("Connecting to WiFi: %s\n", ssid);
    Serial.println("===========================================");
    
    WiFi.mode(WIFI_STA);
    WiFi.setHostname(hostname);
    WiFi.begin(ssid, password);
    
    int attempts = 0;
    while (WiFi.status() != WL_CONNECTED && attempts < 30) {
        delay(500);
        Serial.print(".");
        attempts++;
    }
    
    if (WiFi.status() != WL_CONNECTED) {
        Serial.println("\n❌ WiFi connection failed!");
        return false;
    }
    
    Serial.println("\n✅ WiFi Connected!");
    ip_address = WiFi.localIP().toString();
    
    Serial.println("📡 Network Information:");
    Serial.printf("   IP Address: %s\n", ip_address.c_str());
    Serial.printf("   Signal: %d dBm\n", WiFi.RSSI());
    
    if (MDNS.begin(hostname)) {
        Serial.printf("   mDNS: http://%s.local\n", hostname);
        MDNS.addService("http", "tcp", 80);
        MDNS.addService("ws", "tcp", 81);
    }
    
    Serial.println("===========================================");
    Serial.printf("🌐 Camera Stream: http://%s/stream\n", ip_address.c_str());
    Serial.printf("🔌 WebSocket: ws://%s:81\n", ip_address.c_str());
    Serial.println("===========================================\n");
    
    return true;
}

/* Setup ------------------------------------------------------------------- */
void setup() {
    Serial.begin(115200);
    Serial.setDebugOutput(false);  // Disable debug for less serial clutter
    delay(1000);
    
    Serial.println("\n╔════════════════════════════════════════════╗");
    Serial.println("║  ESP32-CAM Animal Detection (WebSocket)   ║");
    Serial.printf("║  Confidence Threshold: %.0f%%               ║\n", CONFIDENCE_THRESHOLD * 100);
    Serial.println("╚════════════════════════════════════════════╝\n");
    
    // Initialize camera
    Serial.println("📷 Initializing camera...");
    if (!ei_camera_init()) {
        Serial.println("❌ Camera init failed!");
        while(1) delay(1000);
    }
    Serial.println("✅ Camera ready\n");

    // Connect to WiFi
    if (!connectToWiFi()) {
        Serial.println("System halted.");
        while(1) delay(1000);
    }

    // Start HTTP server (for camera stream)
    startCameraServer();
    
    // Start WebSocket server (for inference results)
    webSocket.begin();
    webSocket.onEvent(webSocketEvent);
    Serial.println("✅ WebSocket server started on port 81\n");

    Serial.printf("🧠 Model: %s\n", EI_CLASSIFIER_PROJECT_NAME);
    Serial.printf("📐 Input: %dx%d\n", EI_CLASSIFIER_INPUT_WIDTH, EI_CLASSIFIER_INPUT_HEIGHT);
    Serial.printf("🎯 Threshold: %.0f%%\n\n", CONFIDENCE_THRESHOLD * 100);
    Serial.println("🚀 System ready! Starting inference...\n");
}

/* Main Loop --------------------------------------------------------------- */
void loop() {
    // Handle WebSocket connections
    webSocket.loop();
    
    // Run inference
    if (ei_sleep(200) != EI_IMPULSE_OK) return;

    snapshot_buf = (uint8_t*)malloc(EI_CAMERA_RAW_FRAME_BUFFER_COLS * 
                                    EI_CAMERA_RAW_FRAME_BUFFER_ROWS * 
                                    EI_CAMERA_FRAME_BYTE_SIZE);

    if(snapshot_buf == nullptr) {
        Serial.println("❌ Buffer allocation failed");
        return;
    }

    ei::signal_t signal;
    signal.total_length = EI_CLASSIFIER_INPUT_WIDTH * EI_CLASSIFIER_INPUT_HEIGHT;
    signal.get_data = &ei_camera_get_data;

    if (!ei_camera_capture((size_t)EI_CLASSIFIER_INPUT_WIDTH, 
                          (size_t)EI_CLASSIFIER_INPUT_HEIGHT, 
                          snapshot_buf)) {
        free(snapshot_buf);
        return;
    }

    unsigned long start_time = millis();
    
    ei_impulse_result_t result = { 0 };
    EI_IMPULSE_ERROR err = run_classifier(&signal, &result, false);
    
    if (err != EI_IMPULSE_OK) {
        Serial.printf("❌ Classifier error: %d\n", err);
        free(snapshot_buf);
        return;
    }

    inference_count++;
    int total_time = result.timing.dsp + result.timing.classification + result.timing.anomaly;

    // Find highest confidence
    float max_confidence = 0.0;
    String max_label = "";
    
    for (uint16_t i = 0; i < EI_CLASSIFIER_LABEL_COUNT; i++) {
        if (result.classification[i].value > max_confidence) {
            max_confidence = result.classification[i].value;
            max_label = String(ei_classifier_inferencing_categories[i]);
        }
    }

    // Send result via WebSocket
    if (max_confidence >= MIN_DETECTION_CONFIDENCE) {
        if (max_confidence >= CONFIDENCE_THRESHOLD) {
            valid_detection_count++;
            sendInferenceResult("detected", max_label, max_confidence, total_time);
            Serial.printf("🎯 DETECTED: %s (%.1f%%) - %dms\n", 
                         max_label.c_str(), max_confidence * 100, total_time);
        } else {
            sendInferenceResult("low_confidence", max_label, max_confidence, total_time);
        }
    } else {
        sendInferenceResult("no_animal", "", max_confidence, total_time);
    }

    free(snapshot_buf);
}

/* Camera Functions -------------------------------------------------------- */
bool ei_camera_init(void) {
    if (is_initialised) return true;

    esp_err_t err = esp_camera_init(&camera_config);
    if (err != ESP_OK) {
        Serial.printf("Camera init error: 0x%x\n", err);
        return false;
    }

    sensor_t * s = esp_camera_sensor_get();
    if (s->id.PID == OV3660_PID) {
        s->set_vflip(s, 1);
        s->set_brightness(s, 1);
        s->set_saturation(s, 0);
    }

    is_initialised = true;
    return true;
}

bool ei_camera_capture(uint32_t img_width, uint32_t img_height, uint8_t *out_buf) {
    if (!is_initialised) return false;

    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb) return false;

    bool converted = fmt2rgb888(fb->buf, fb->len, PIXFORMAT_JPEG, snapshot_buf);
    esp_camera_fb_return(fb);

    if(!converted) return false;

    if ((img_width != EI_CAMERA_RAW_FRAME_BUFFER_COLS) || 
        (img_height != EI_CAMERA_RAW_FRAME_BUFFER_ROWS)) {
        ei::image::processing::crop_and_interpolate_rgb888(
            out_buf, EI_CAMERA_RAW_FRAME_BUFFER_COLS, EI_CAMERA_RAW_FRAME_BUFFER_ROWS,
            out_buf, img_width, img_height);
    }

    return true;
}

static int ei_camera_get_data(size_t offset, size_t length, float *out_ptr) {
    size_t pixel_ix = offset * 3;
    size_t out_ptr_ix = 0;

    for (size_t i = 0; i < length; i++) {
        out_ptr[out_ptr_ix] = (snapshot_buf[pixel_ix + 2] << 16) + 
                              (snapshot_buf[pixel_ix + 1] << 8) + 
                              snapshot_buf[pixel_ix];
        out_ptr_ix++;
        pixel_ix += 3;
    }
    return 0;
}

#if !defined(EI_CLASSIFIER_SENSOR) || EI_CLASSIFIER_SENSOR != EI_CLASSIFIER_SENSOR_CAMERA
#error "Invalid model for current sensor"
#endif