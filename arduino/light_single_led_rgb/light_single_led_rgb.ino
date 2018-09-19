// NeoPixel Ring simple sketch (c) 2013 Shae Erisson
// released under the GPLv3 license to match the rest of the AdaFruit NeoPixel library

#include <Adafruit_NeoPixel.h>
#ifdef __AVR__
  #include <avr/power.h>
#endif

// Which pin on the Arduino is connected to the NeoPixels?
// On a Trinket or Gemma we suggest changing this to 1
#define PIN            6

// How many NeoPixels are attached to the Arduino?
#define NUMPIXELS      255

// When we setup the NeoPixel library, we tell it how many pixels, and which pin to use to send signals.
// Note that for older NeoPixel strips you might need to change the third parameter--see the strandtest
// example for more information on possible values.
Adafruit_NeoPixel pixels = Adafruit_NeoPixel(NUMPIXELS, PIN, NEO_GRB + NEO_KHZ800);

uint8_t current_led = 255; // 255 means no pixel
uint8_t current_r = 0;
uint8_t current_g = 0;
uint8_t current_b = 0;

void setup() {
  // This is for Trinket 5V 16MHz, you can remove these three lines if you are not using a Trinket
#if defined (__AVR_ATtiny85__)
  if (F_CPU == 16000000) clock_prescale_set(clock_div_1);
#endif
  // End of trinket special code

  pixels.begin(); // This initializes the NeoPixel library.
  Serial.begin(115200);
  //Serial.begin(9600);
  pixels.show();
}

void loop() {

  while (Serial.available() == 0){
  }
  uint8_t led = Serial.read();
  while (Serial.available() == 0){
  }
  uint8_t r = Serial.read();
  while (Serial.available() == 0){
  }
  uint8_t g = Serial.read();
  while (Serial.available() == 0){
  }
  uint8_t b = Serial.read();
  
  if (led != current_led || r != current_r || g != current_g || b != current_b){
    if (led < NUMPIXELS){
      pixels.setPixelColor(led, pixels.Color(r, g, b));
    }
    if (led != current_led){
      pixels.setPixelColor(current_led, 0);
    }
    pixels.show(); // This sends the updated pixel color to the hardware.
    //delay(30); // delay so the next command can't disrupt the chain (i suppose that happens if you send too quickly)
    current_led = led;
    current_r = r;
    current_g = g;
    current_b = b;
  }

   // writing this is the ready signal for sending more bytes
  Serial.write(1);
  
}
