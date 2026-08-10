/*
 * burgerbot motor controller -- Raspberry Pi Pico (RP2040)
 * ========================================================
 *
 * Closes a velocity loop on both drive wheels and talks to the Pi over USB
 * CDC. The Pi runs ros2_control; this board runs the only thing that genuinely
 * needs hard real time.
 *
 * Build: Arduino IDE with the earlephilhower "Raspberry Pi Pico/RP2040" core.
 *   Board: Raspberry Pi Pico
 *   Then hold BOOTSEL, plug in, and Upload. Afterwards it appears on the Pi as
 *   /dev/ttyACM0 (an Arduino would have been /dev/ttyUSB0 -- see the udev rule
 *   in the repo README).
 *
 * !!! 3.3 V LOGIC !!!
 * The Pico is a 3.3 V part and its GPIOs are NOT 5 V tolerant, unlike the
 * Arduino Uno this design came from. Before powering anything up:
 *   - Encoder outputs must be 3.3 V. Many geared motors ship with 5 V hall
 *     encoders; those need a level shifter or a divider on A and B, or they
 *     will damage the Pico.
 *   - L298N / TB6612 inputs are fine driven at 3.3 V.
 *   - Never back-feed motor supply into VSYS without a diode.
 *
 * Serial protocol (unchanged from the course, so its debugging still applies,
 * and readable in any serial monitor):
 *
 *   Pi  -> Pico   "r<s>NN.NN,l<s>NN.NN,"   target wheel speed, rad/s
 *   Pico -> Pi    "r<s>NN.NN,l<s>NN.NN,"   measured wheel speed, rad/s
 *
 * where <s> is 'p' or 'n' for sign. ~30 bytes at 50 Hz is nothing for USB CDC.
 *
 * Differences from the Arduino original, all deliberate:
 *   - Full 4x quadrature decode on both channels and both edges, giving four
 *     times the resolution and immunity to the direction glitches you get from
 *     sampling phase B inside a single-edge interrupt.
 *   - Signed velocity into the PID. The original fed it |velocity|, so the
 *     controller could not tell forward from reverse and fought itself the
 *     moment a wheel was commanded to change direction.
 *   - 100 Hz control loop instead of 10 Hz.
 *   - A command timeout. If the Pi stops talking -- crash, unplugged cable,
 *     killed node -- the motors stop. The original would happily keep driving
 *     into a wall forever.
 */

#include <Arduino.h>

// ---------------------------------------------------------------- pins ----
// Motor driver (L298N-style: one PWM + two direction lines per channel).
const uint8_t PIN_R_PWM = 16;
const uint8_t PIN_R_IN1 = 17;
const uint8_t PIN_R_IN2 = 18;
const uint8_t PIN_L_PWM = 19;
const uint8_t PIN_L_IN1 = 20;
const uint8_t PIN_L_IN2 = 21;

// Quadrature encoders. Both channels are interrupt sources.
const uint8_t PIN_R_ENC_A = 10;
const uint8_t PIN_R_ENC_B = 11;
const uint8_t PIN_L_ENC_A = 12;
const uint8_t PIN_L_ENC_B = 13;

// ------------------------------------------------------------ calibrate ----
// Counts per output-shaft revolution with 4x decoding.
//
// CALIBRATE THIS. Mark a wheel, turn it exactly ten revolutions by hand, and
// read the reported position; divide by ten. Everything downstream -- odometry,
// the EKF, AMCL, the costmap -- inherits any error here, and a wheel constant
// that is 5% wrong will send the robot in a slow curve when told to go
// straight. The default is the course's 385 single-edge counts x 4.
const float ENCODER_TICKS_PER_REV = 1540.0f;

// Flip if a wheel reports negative speed when driven forward.
const int8_t R_ENCODER_SIGN = +1;
const int8_t L_ENCODER_SIGN = +1;
// Flip if a wheel physically spins backwards on a positive command.
const int8_t R_MOTOR_SIGN = +1;
const int8_t L_MOTOR_SIGN = +1;

const uint32_t CONTROL_PERIOD_US = 10000;  // 100 Hz
const uint32_t REPORT_PERIOD_US = 20000;   // 50 Hz upstream

// Stop if the Pi goes quiet for this long.
const uint32_t COMMAND_TIMEOUT_US = 500000;  // 0.5 s

// Below this the wheel cannot turn at all; holding integral here just winds up.
const float MIN_COMMAND_RAD_S = 0.05f;

const int PWM_MAX = 255;

// ------------------------------------------------------------- encoders ----
// Written from ISRs, read from the loop.
volatile int32_t r_ticks = 0;
volatile int32_t l_ticks = 0;
volatile uint8_t r_state = 0;
volatile uint8_t l_state = 0;

// Full quadrature state table indexed by (previous << 2) | current, where each
// state is (A << 1) | B. Values are the tick delta; 0 entries are no-change,
// and the 2s are impossible double transitions that mean a missed edge -- they
// are counted as 0 rather than guessed at.
const int8_t QUAD_TABLE[16] = {
  0, +1, -1, 0,
  -1, 0, 0, +1,
  +1, 0, 0, -1,
  0, -1, +1, 0
};

static inline uint8_t readState(uint8_t pinA, uint8_t pinB) {
  return (uint8_t)((digitalRead(pinA) << 1) | digitalRead(pinB));
}

void rightEncoderISR() {
  uint8_t s = readState(PIN_R_ENC_A, PIN_R_ENC_B);
  r_ticks += QUAD_TABLE[(r_state << 2) | s];
  r_state = s;
}

void leftEncoderISR() {
  uint8_t s = readState(PIN_L_ENC_A, PIN_L_ENC_B);
  l_ticks += QUAD_TABLE[(l_state << 2) | s];
  l_state = s;
}

// ------------------------------------------------------------------ PID ----
struct Pid {
  float kp, ki, kd;
  float integral = 0.0f;
  float prev_error = 0.0f;

  float update(float setpoint, float measured, float dt) {
    const float error = setpoint - measured;
    const float derivative = (dt > 1e-6f) ? (error - prev_error) / dt : 0.0f;
    prev_error = error;

    float unclamped = kp * error + ki * (integral + error * dt) + kd * derivative;

    // Conditional integration: only accumulate when the output is not already
    // saturated, or when integrating would pull it back into range. Without
    // this the integral charges up during any sustained saturation -- stalled
    // wheel, steep carpet -- and the motor then overshoots wildly when it
    // finally frees. This is the fix the PID_v1 default arrangement lacks.
    const bool saturated = (unclamped > PWM_MAX) || (unclamped < -PWM_MAX);
    const bool winding_down = (unclamped > 0.0f) != (error > 0.0f);
    if (!saturated || winding_down) {
      integral += error * dt;
    }

    float out = kp * error + ki * integral + kd * derivative;
    return constrain(out, (float)-PWM_MAX, (float)PWM_MAX);
  }

  void reset() {
    integral = 0.0f;
    prev_error = 0.0f;
  }
};

// Starting point from the course's tuning. Expect to retune: the Pico's loop
// is ten times faster, so the same gains behave differently.
Pid right_pid{ 11.5f, 7.5f, 0.10f };
Pid left_pid{ 12.8f, 8.3f, 0.10f };

// -------------------------------------------------------------- command ----
float r_cmd_vel = 0.0f;  // rad/s, signed
float l_cmd_vel = 0.0f;
float r_meas_vel = 0.0f;
float l_meas_vel = 0.0f;

uint32_t last_command_us = 0;
uint32_t last_control_us = 0;
uint32_t last_report_us = 0;
int32_t r_ticks_prev = 0;
int32_t l_ticks_prev = 0;

// Incremental parser for "r<s>NN.NN,l<s>NN.NN,".
char parse_field[12];
uint8_t parse_len = 0;
char parse_wheel = 0;
int8_t parse_sign = 1;

void applyMotor(uint8_t pwm_pin, uint8_t in1, uint8_t in2, int8_t motor_sign, float output) {
  const float signed_out = output * motor_sign;
  const int magnitude = (int)constrain(fabsf(signed_out), 0.0f, (float)PWM_MAX);

  if (magnitude == 0) {
    // Both low = coast. Braking on every zero command makes the robot jerk at
    // the end of each path segment, which the navigation stack then has to
    // fight.
    digitalWrite(in1, LOW);
    digitalWrite(in2, LOW);
    analogWrite(pwm_pin, 0);
    return;
  }
  digitalWrite(in1, signed_out >= 0.0f ? HIGH : LOW);
  digitalWrite(in2, signed_out >= 0.0f ? LOW : HIGH);
  analogWrite(pwm_pin, magnitude);
}

void handleChar(char c) {
  if (c == 'r' || c == 'l') {
    parse_wheel = c;
    parse_len = 0;
    parse_sign = 1;
  } else if (c == 'p') {
    parse_sign = 1;
  } else if (c == 'n') {
    parse_sign = -1;
  } else if (c == ',' || c == '\n' || c == '\r') {
    if (parse_wheel && parse_len > 0) {
      parse_field[parse_len] = '\0';
      const float value = parse_sign * atof(parse_field);
      if (parse_wheel == 'r') {
        r_cmd_vel = value;
      } else {
        l_cmd_vel = value;
      }
      last_command_us = micros();
    }
    parse_wheel = 0;
    parse_len = 0;
  } else if ((c >= '0' && c <= '9') || c == '.') {
    if (parse_len < sizeof(parse_field) - 1) {
      parse_field[parse_len++] = c;
    }
  }
  // Anything else is noise; drop it rather than corrupting the field.
}

void reportVelocities() {
  // Magnitude plus an explicit sign character, matching what the Pi's
  // BurgerbotInterface parses.
  char buf[48];
  snprintf(buf, sizeof(buf), "r%c%05.2f,l%c%05.2f,",
           r_meas_vel >= 0.0f ? 'p' : 'n', (double)fabsf(r_meas_vel),
           l_meas_vel >= 0.0f ? 'p' : 'n', (double)fabsf(l_meas_vel));
  Serial.println(buf);
}

void setup() {
  pinMode(PIN_R_PWM, OUTPUT);
  pinMode(PIN_R_IN1, OUTPUT);
  pinMode(PIN_R_IN2, OUTPUT);
  pinMode(PIN_L_PWM, OUTPUT);
  pinMode(PIN_L_IN1, OUTPUT);
  pinMode(PIN_L_IN2, OUTPUT);
  applyMotor(PIN_R_PWM, PIN_R_IN1, PIN_R_IN2, R_MOTOR_SIGN, 0.0f);
  applyMotor(PIN_L_PWM, PIN_L_IN1, PIN_L_IN2, L_MOTOR_SIGN, 0.0f);

  // 20 kHz keeps the PWM whine out of the audible band. The default ~500 Hz
  // makes a small robot sing loudly enough to be genuinely annoying indoors.
  analogWriteFreq(20000);
  analogWriteRange(PWM_MAX);

  // INPUT_PULLUP suits open-collector hall encoders and is harmless for
  // push-pull ones.
  pinMode(PIN_R_ENC_A, INPUT_PULLUP);
  pinMode(PIN_R_ENC_B, INPUT_PULLUP);
  pinMode(PIN_L_ENC_A, INPUT_PULLUP);
  pinMode(PIN_L_ENC_B, INPUT_PULLUP);

  r_state = readState(PIN_R_ENC_A, PIN_R_ENC_B);
  l_state = readState(PIN_L_ENC_A, PIN_L_ENC_B);

  attachInterrupt(digitalPinToInterrupt(PIN_R_ENC_A), rightEncoderISR, CHANGE);
  attachInterrupt(digitalPinToInterrupt(PIN_R_ENC_B), rightEncoderISR, CHANGE);
  attachInterrupt(digitalPinToInterrupt(PIN_L_ENC_A), leftEncoderISR, CHANGE);
  attachInterrupt(digitalPinToInterrupt(PIN_L_ENC_B), leftEncoderISR, CHANGE);

  Serial.begin(115200);  // Rate is nominal on USB CDC.

  const uint32_t now = micros();
  last_control_us = now;
  last_report_us = now;
  last_command_us = now;
}

void loop() {
  while (Serial.available()) {
    handleChar((char)Serial.read());
  }

  const uint32_t now = micros();
  if ((uint32_t)(now - last_control_us) < CONTROL_PERIOD_US) {
    return;
  }
  const float dt = (now - last_control_us) * 1e-6f;
  last_control_us = now;

  // Snapshot the ISR counters. 32-bit reads are not atomic against an
  // interrupt on Cortex-M0+, so guard them.
  noInterrupts();
  const int32_t r_now = r_ticks;
  const int32_t l_now = l_ticks;
  interrupts();

  const int32_t r_delta = r_now - r_ticks_prev;
  const int32_t l_delta = l_now - l_ticks_prev;
  r_ticks_prev = r_now;
  l_ticks_prev = l_now;

  const float rad_per_tick = TWO_PI / ENCODER_TICKS_PER_REV;
  const float r_raw = R_ENCODER_SIGN * r_delta * rad_per_tick / dt;
  const float l_raw = L_ENCODER_SIGN * l_delta * rad_per_tick / dt;

  // Light low-pass. Tick quantisation makes raw velocity noisy at low speed,
  // and that noise goes straight into the D term.
  const float alpha = 0.30f;
  r_meas_vel += alpha * (r_raw - r_meas_vel);
  l_meas_vel += alpha * (l_raw - l_meas_vel);

  const bool stale = (uint32_t)(now - last_command_us) > COMMAND_TIMEOUT_US;
  if (stale) {
    r_cmd_vel = 0.0f;
    l_cmd_vel = 0.0f;
  }

  float r_out, l_out;
  if (fabsf(r_cmd_vel) < MIN_COMMAND_RAD_S) {
    right_pid.reset();
    r_out = 0.0f;
  } else {
    r_out = right_pid.update(r_cmd_vel, r_meas_vel, dt);
  }
  if (fabsf(l_cmd_vel) < MIN_COMMAND_RAD_S) {
    left_pid.reset();
    l_out = 0.0f;
  } else {
    l_out = left_pid.update(l_cmd_vel, l_meas_vel, dt);
  }

  applyMotor(PIN_R_PWM, PIN_R_IN1, PIN_R_IN2, R_MOTOR_SIGN, r_out);
  applyMotor(PIN_L_PWM, PIN_L_IN1, PIN_L_IN2, L_MOTOR_SIGN, l_out);

  if ((uint32_t)(now - last_report_us) >= REPORT_PERIOD_US) {
    last_report_us = now;
    reportVelocities();
  }
}
