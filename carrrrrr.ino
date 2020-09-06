#include <MFRC522.h>  // MFRC522 RFID module library.
#include <SPI.h>      // SPI device communication library.
#include <EEPROM.h>   // EEPROM (memory) library.

#define pinRST 9      // Defines pins for RST, SS conncetions respectively.
#define pinSS 4

byte readCard[4];     // Array that will hold UID of the RFID card.
int successRead;
//byte inChar;

MFRC522 mfrc522(pinSS, pinRST);   // Creates MFRC522 instance.
MFRC522::MIFARE_Key key;          // Creates MIFARE key instance.
char t;
int forwardright = 3;
int forwardleft =10;
int backwardright = 5;
int backwardleft = 6;

const int trigPin = 8;
const int echoPin = 7;
long duration;
int distance;

void setup() {
 pinMode(trigPin, OUTPUT); // Sets the trigPin as an Output
pinMode(echoPin, INPUT); // Sets the echoPin as an Input
Serial.begin(9600);

pinMode(forwardright,OUTPUT);   //right motors forward
pinMode(forwardleft,OUTPUT);   //left motors forward
pinMode(backwardright,OUTPUT);   //right motors reverse
pinMode(backwardleft,OUTPUT);   //right motors reverse
 
 SPI.begin();        // Initiates SPI connection between RFID module and Arduino.
 
  mfrc522.PCD_Init(); // Initiates MFRC522 RFID module.

  do {
    successRead = getID();     // Loops getID library function until reading process is done.
  }

  while (!successRead);
  for ( int i = 0; i < mfrc522.uid.size; i++ )  // You can add multiple cards to read in the for loop.
  {
    EEPROM.write(i, readCard[i] );     // Saves RFID cards UID to EEPROM.
  }
}

void loop() {

getID();

digitalWrite(trigPin, LOW);
delayMicroseconds(2);
// Sets the trigPin on HIGH state for 10 micro seconds
digitalWrite(trigPin, HIGH);
delayMicroseconds(10);
digitalWrite(trigPin, LOW);
// Reads the echoPin, returns the sound wave travel time in microseconds
duration = pulseIn(echoPin, HIGH);
// Calculating the distance
distance= duration*0.034/2;
  
  
if(Serial.available()){
  t = Serial.read();
  if(t == 1){Serial.println("Automatic control");}
     Stop(); 
    switch(t){
    case '1':  
    if(distance < 30){
      left();
      delay(1000);
      forward();
      delay(200);
      right();
      delay(1500);
      break;
      }else{
      forward();
      break;
      }
    case '2':  
    if(distance < 30){
      left();
      delay(1000);
      forward();
      delay(200);
      right();
      delay(1500);
      break;
      }else{
       back();
      break;
      }
    case '3':
    if(distance < 30){
      left();
      delay(1000);
      forward();
      delay(200);
      right();
      delay(1500);
      break;
      }else{  
      left();
      break;
      }
    case '4':
    if(distance < 30){
      left();
      delay(1000);
      forward();
      delay(200);
      right();
      delay(1500);
      break;
      }else{
      right();
      break;
      }
    }
  } 
}


int getID() // Function that will read and print the RFID cards UID.
{
  if ( ! mfrc522.PICC_IsNewCardPresent())  // If statement that looks for new cards.
  {
    return;
  }

  if ( ! mfrc522.PICC_ReadCardSerial())    // If statement that selects one of the cards.
  {
    return;
  }
 
  
  for (int i = 0; i < mfrc522.uid.size; i++) {  
    readCard[i] = mfrc522.uid.uidByte[i];   // Reads RFID cards UID.
    Serial.print(readCard[i], HEX);
    // Prints RFID cards UID to the serial monitor.
    //then send it 2 mobile
  }
  Serial.print("\n");     

  mfrc522.PICC_HaltA();     // Stops the reading process.
  delay(400);
}

void forward()
{
 analogWrite(forwardright,65);
  analogWrite(backwardright,0);
  analogWrite(backwardleft,0);
  analogWrite(forwardleft,65);
}


void back()
{
 analogWrite(forwardright,0);
  analogWrite(backwardright,65);
  analogWrite(backwardleft,65);
  analogWrite(forwardleft,0);
}

void left()
{
  analogWrite(forwardright,60);
  analogWrite(backwardright,0);
  analogWrite(backwardleft,60);
  analogWrite(forwardleft,0);
}

void right()
{
   analogWrite(forwardright,0);
  analogWrite(backwardright,60);
  analogWrite(backwardleft,0);
  analogWrite(forwardleft,60);
}
void Stop()
{
   digitalWrite(backwardleft,LOW);
  digitalWrite(forwardleft,LOW);
  digitalWrite(forwardright,LOW);
  digitalWrite(backwardright,LOW);
}
