use serialport;
use std::time::Duration;


fn main() {

    let mut serial_port = serialport::new("/dev/ttyACM0", 57600)
        .timeout(Duration::from_secs(5))
        .open()
        .expect("Failed just like you");

    println!("Connection established");

    loop {
        let mut input = String::new();
        let stdin = std::io::stdin();
        stdin.read_line(&mut input).unwrap();

        println!("{}", input);
        
        let direction = input.as_bytes();

        serial_port.write(direction).expect("Write Failed!");
        serial_port.flush().unwrap();
    }


}
