// Simple strand test for Adafruit Dot Star RGB LED strip.
// This is a basic diagnostic tool, NOT a graphics demo...helps confirm
// correct wiring and tests each pixel's ability to display red, green
// and blue and to forward data down the line.  By limiting the number
// and color of LEDs, it's reasonably safe to power a couple meters off
// the Arduino's 5V pin.  DON'T try that with other code!

#include <Adafruit_DotStar.h>
// Because conditional #includes don't work w/Arduino sketches...
#include <SPI.h>         // COMMENT OUT THIS LINE FOR GEMMA OR TRINKET
//#include <avr/power.h> // ENABLE THIS LINE FOR GEMMA OR TRINKET

#define NUMPIXELS 10 // Number of LEDs in strip

// Here's how to control the LEDs from any two pins:
//#define DATAPIN    4
//#define CLOCKPIN   5
//Adafruit_DotStar strip = Adafruit_DotStar(
//  NUMPIXELS, DATAPIN, CLOCKPIN, DOTSTAR_BRG);
// The last parameter is optional -- this is the color data order of the
// DotStar strip, which has changed over time in different production runs.
// Your code just uses R,G,B colors, the library then reassigns as needed.
// Default is DOTSTAR_BRG, so change this if you have an earlier strip.

// Hardware SPI is a little faster, but must be wired to specific pins
// (Arduino Uno = pin 11 for data, 13 for clock, other boards are different).
Adafruit_DotStar strip = Adafruit_DotStar(NUMPIXELS, DOTSTAR_BRG);

uint8_t current_led = 255; // 255 means all pixels
uint8_t current_r = 0;
uint8_t current_g = 0;
uint8_t current_b = 0;

void setup() {

#if defined(__AVR_ATtiny85__) && (F_CPU == 16000000L)
  clock_prescale_set(clock_div_1); // Enable 16 MHz on Trinket
#endif

  strip.begin(); // This initializes the NeoPixel library.
  Serial.begin(115200);
  //Serial.begin(9600);
  strip.show();
}

// Runs 10 LEDs at a time along strip, cycling through red, green and blue.
// This requires about 200 mA for all the 'on' pixels + 1 mA per 'off' pixel.

int      head  = 0, tail = -10; // Index of first 'on' and 'off' pixels
uint32_t color = 0xFF0000;      // 'On' color (starts red)

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
        
        strip.setPixelColor(i, strip.Color(r, g, b));
        
      }
    }
    
    // light a single led
    else {
      
      // previously all leds were lit
      if (current_led == NUMPIXELS){
        
        for (int i=0; i < NUMPIXELS; i++){

          // light the led
          if (i == led){
            strip.setPixelColor(i, strip.Color(r, g, b));
          }
          
          // turn off all leds but the chosen one
          else {
            strip.setPixelColor(i, 0);
          }
        }
      }
      
      // previously only one led was lit
      else {
        
        // light single led
        strip.setPixelColor(led, strip.Color(r, g, b));
        
        // turn off previous led
        if (led != current_led){
          strip.setPixelColor(current_led, 0);
        }
      }
      
    }
       
    strip.show(); // This sends the updated pixel color to the hardware.
    current_led = led;
    current_r = r;
    current_g = g;
    current_b = b;
  }

   // writing this is the ready signal for sending more bytes
  Serial.write(1);
  
}
