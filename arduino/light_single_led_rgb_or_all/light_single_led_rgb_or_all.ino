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

uint8_t current_led = 255; // 255 means all pixels
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

  // something must have changed compared to the previous command to execute a new one
  if (led != current_led || r != current_r || g != current_g || b != current_b){

    // light all leds with the same color
    if (led == NUMPIXELS){
      
      for (int i=0; i < NUMPIXELS; i++){
        
        pixels.setPixelColor(i, pixels.Color(r, g, b));
        
      }
    }
    
    // light a single led
    else {
      
      // previously all leds were lit
      if (current_led == NUMPIXELS){
        
        for (int i=0; i < NUMPIXELS; i++){

          // light the led
          if (i == led){
            pixels.setPixelColor(i, pixels.Color(r, g, b));
          }
          
          // turn off all leds but the chosen one
          else {
            pixels.setPixelColor(i, 0);
          }
        }
      }
      
      // previously only one led was lit
      else {
        
        // light single led
        pixels.setPixelColor(led, pixels.Color(r, g, b));
        
        // turn off previous led
        if (led != current_led){
          pixels.setPixelColor(current_led, 0);
        }
      }
      
    }
       
    pixels.show(); // This sends the updated pixel color to the hardware.
    current_led = led;
    current_r = r;
    current_g = g;
    current_b = b;
  }

   // writing this is the ready signal for sending more bytes
  Serial.write(1);
  
}
