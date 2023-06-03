#![no_std]
#![no_main]

use arduino_hal::prelude::*;
use panic_halt as _;

use embedded_hal::serial::Read;

#[arduino_hal::entry]
fn main() -> ! {
    let dp = arduino_hal::Peripherals::take().unwrap();
    let pins = arduino_hal::pins!(dp);
    let mut serial = arduino_hal::default_serial!(dp, pins, 57600);

    // led to check
    let mut led2 = pins.d2.into_output(); // Front Right
    let mut led3 = pins.d3.into_output(); // Front Left
    let mut led4 = pins.d4.into_output(); // Back Right
    let mut led5 = pins.d5.into_output(); // Back Left


    ufmt::uwriteln!(&mut serial, "Hello from Arduino!\r").void_unwrap();

    loop {
        // Read a byte from the serial connection
        let b = nb::block!(serial.read()).void_unwrap();

        // Car standing still

        led2.set_low();
        led3.set_low();
        led4.set_low();
        led5.set_low();


        if b == 49 { // 1 Forward, Back wheels 
            led4.toggle();
            led5.toggle();
        }
        if b == 50 { // 2 Left
            led3.toggle();
            led5.toggle();
        }
        if b == 51 { // 3 Right
            led2.toggle();
            led4.toggle();
        }
        if b == 52 {
            led2.toggle();
            led3.toggle();
            led4.toggle();
            led5.toggle();
        }

        arduino_hal::delay_ms(2000);
    }
}
