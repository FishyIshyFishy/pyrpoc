# Controlling the Coherent OBIS Laser Box

This is the hardware reference for driving the rig's OBIS LX/LS Laser Box from pyrpoc. It covers
what Coherent provides, how the box talks, the commands and replies, the safety chain, and what
still has to be checked on the rig. A separate session uses it to write the device. How the device
fits into pyrpoc (inputs, panels, programs) is that session's job. Here it is limited to the
hardware facts that shape it (section 10).

**Nothing in this document has been run against the hardware.** It was written without access to
the box, and Coherent's site (and with it the operator's manual) could not be reached from where it
was written. Every fact carries its source, so you know how far to trust it:

| Mark | Meaning |
|---|---|
| **seen** | Shown in the rig's own photos of the box and of Coherent Connection (Nov 2026). |
| **driver** | Used by an open-source driver that ran on real OBIS hardware: [amsikking's Laser Box adaptor][amsikking] (tested on the same 5-bay box, SKU 1343229), [Micro-Manager's OBIS adapters][mm], or [SWC's OBISLaser][swc] (whose rig log records the replies). |
| **verify** | Recalled from Coherent's *OBIS LX/LS Operator's Manual* or inferred, and not confirmed. Check it on the rig or in the manual before relying on it. |

When a fact is confirmed on the rig, change its mark to **rig** with the date.

## 1. The hardware on this rig

From the photos (**seen**):

- One **OBIS LX/LS Laser Box** (5 bays) reached over USB as **`COM8`**. Coherent Connection lists
  its lasers as `COM8:1` … `COM8:5`: one port, one channel per bay.
- Front panel: DC in, USB, power button, a **key switch** (STANDBY/ON), a box status LED, an
  **interlock** terminal (green 2-pin block, closed with a jumper wire), an **RS-232** DB9, and per
  laser an **ANALOG** and a **DIGITAL** SMB modulation input with a status LED.
- **All five DIGITAL inputs are cabled.** The ANALOG inputs are not. So the lasers are gated by
  TTL from some other instrument (probably the NI-DAQ's digital lines). Before integrating, write
  down which source drives which bay.

| Channel | Laser | Type (inferred) | Setpoint now | Mode now |
|---|---|---|---|---|
| 1 | 375-25FP (LX 375 nm, 25 mW) | diode (DDL) | 6.0 mW | Digital:Current |
| 2 | 405-50FP (LX 405 nm, 50 mW) | diode (DDL) | 1.0 mW | Digital:Current |
| 3 | 488-30FP (LX 488 nm, 30 mW) | diode (DDL) | 5.0 mW | Digital:Current |
| 4 | OBIS LS 561-60 (LS 561 nm, 60 mW) | optically pumped (OPSL) | 1.0 mW | "Digital Modulation" |
| 5 | 594-60FP (LS 594 nm, 60 mW) | optically pumped (OPSL) | 1.0 mW | "Digital Modulation" |

The type column is inferred from LX (diode) vs LS (OPSL) and from the shorter mode name Coherent
Connection shows for 4 and 5. `SYSTem<n>:INFormation:TYPe?` answers it (section 4). It matters
because the two types accept different modes and different minimum powers (sections 5 and 6).

Coherent Connection showed **"0.0 mW Emitting"** for every laser with each one started. In digital
mode that is what a laser shows while its TTL is low: it is enabled, and the TTL gates the light.

## 2. What Coherent provides, and what to use

Coherent has no standalone OBIS SDK for Python. There are four ways in:

1. **Text commands over the serial port (use this).** Each OBIS laser, and the Laser Box, speaks
   a SCPI-style ASCII command set over a serial link. Over USB the box appears as a virtual COM
   port (`COM8`), and the RS-232 DB9 carries the same commands. Coherent's operator's manual
   documents this command set. Every open-source driver above uses it. In Python it needs only
   [`pyserial`][pyserial], which would be a new dependency in `pyproject.toml`.
2. **`CoherentConnectionLibrary.dll`** (**driver**, per SWC). This is the .NET assembly inside
   Coherent Connection 6. `ConnectionLibrary.SerialCommunicator` is a serial line transport, and
   `ConnectionLibrary.OBIS_Laser` has `Start`, `Stop`, `PowerLevel_AsReal_mW` and
   `OperatingMode_AsEnum`. Don't use it. It is undocumented and internal to Coherent's app. From
   Python it would need `pythonnet` and the .NET runtime, and it may not be redistributable. Under
   it are the same serial commands, so it adds nothing but a layer we can't see into or test.
3. **Coherent's LabVIEW drivers.** These are not usable from Python.
4. **Coherent Connection** (the GUI in the photos). It is useful on the rig for checking and for
   reading the manual's status texts. It is not a control path.

Two practical consequences of the GUI:

- **Coherent Connection holds the port while it runs.** Close it before pyrpoc connects, and
  don't run both. The USB driver installed with Coherent Connection is what makes `COM8` appear,
  so keep it installed.
- **Coherent Connection probes every COM port on the PC** while looking for lasers. That can upset
  other serial devices. On this rig that includes the Prior and Zaber stages. amsikking advises
  adding the other ports to its *device exclusion list*. Do that on the rig PC.

## 3. The serial link

| Fact | Value | Source |
|---|---|---|
| Port settings | 115200 baud, 8 data bits, no parity, 1 stop bit, no flow control | **driver** (115200 in all three; rest **verify**) |
| Command terminator | `\r` (CR). CR+LF is also accepted. | **driver** (amsikking sends `\r`; SWC sends `\r\n`) |
| Reply terminator | `\r\n` | **driver** |
| Encoding | ASCII | **driver** |
| Case | keywords are case-insensitive; the short form is the capitals (`SOURce` → `SOUR`) | **verify** |

### Handshaking: how a reply is framed

With **handshaking on**, which is the state amsikking found the box in and the state SWC and
Micro-Manager set at connect (**driver**):

- A **command** (a setting) replies with one line, `OK`.
- A **query** (ends in `?`) replies with its value line, then `OK`.
- A **refused** command or query replies with one line, `ERR-<code>`, and **no** `OK` follows.
  amsikking raises on the `ERR` line without reading another, and its framing assertion
  (`in_waiting == 0`) held on the real box.

With handshaking off, commands reply with nothing and queries with the value only (**verify**).
That makes errors invisible and framing guesswork. The driver should therefore turn it on at
connect (`SYSTem<n>:COMMunicate:HANDshaking ON`) and read every exchange to its `OK` or `ERR`.

There is also a *prompt* setting, `SYSTem<n>:COMMunicate:PROMpt ON|OFF`, which adds a prompt text
to replies. Micro-Manager turns it OFF at connect (**driver**). The probe in section 11 shows what
the box is set to.

**Don't purge the input before every command,** as Micro-Manager does. That hides a framing bug
instead of reporting it. Purge once at connect, then treat any line you didn't expect as an error.

### Error codes

From amsikking's table, which ran on this box model (**driver**):

| Reply | Meaning |
|---|---|
| `ERR-000` | No error |
| `ERR-100` | Unrecognized command/query |
| `ERR-102` | Syntax error |
| `ERR-109` | Parameter missing |
| `ERR-200` | Execution error |
| `ERR-203` | Command protected |
| `ERR-220` | Invalid parameter |
| `ERR-221` | Settings conflict |
| `ERR-241` | Device unavailable (e.g. an empty bay, or the box asked a laser-only question) |
| `ERR-256` / `ERR-257` | File does not exist / file to open not named |
| `ERR-310` | System error |
| `ERR-321` | Out of memory |
| `ERR-350` | Queue overflow |
| `ERR-400` | Query unavailable |
| `ERR-500` | CCB fault |
| `ERR-510` | I2C bus fault |
| `ERR-520` | Controller time out |
| `ERR-900` | CCB message timed out |

The driver should name these in its error message, and should pass on an unknown code as the
number rather than guessing at it.

### Addressing the box and its lasers

One port reaches the box and all five lasers. A **channel number** on the command's first keyword
chooses which one (**driver**, **seen** as `COM8:n`):

- **Channel 0** is the Laser Box itself, the controller. It has no wavelength: asking it
  `SYSTem0:INFormation:WAVelength?` gives `ERR-100` or `ERR-241`.
- **Channels 1–5** are the bays. An empty bay answers `ERR-241` (*Device unavailable*).
- The number goes on the root keyword: `SYSTem3:INFormation:WAVelength?`, `SOURce3:AM:STATe ON`,
  `*IDN3?`. amsikking used long forms (`SOURce%i:POWer:LEVel:IMMediate:AMPLitude`) and
  Micro-Manager used short ones (`SOUR%i:POW:LEV:IMM:AMPL`). Both worked on real hardware.

**Always send the channel number.** Without one, the target is unclear: on SWC's OBIS Remote,
`*IDN?` and `MODel?` were answered by the controller, but `WAVelength?` was answered by the laser
(**driver**).

**Prefer the short forms.** SWC's OBIS Remote (firmware v2.0.2) refused one long form with
`ERR-100` and accepted its short form `SYST:COMM:HAND` (**driver**). That long form was misspelled
(`HANDshake` for `HANDshaking`), so this may only be a spelling problem. Still, the short forms are
what Micro-Manager sent, and they cost nothing.

## 4. Command reference

`<n>` is the channel (section 3). Powers are in **watts** on the wire. The device converts to and
from mW at its boundary.

### Identity and limits (read at connect)

| Command | Reply | Source |
|---|---|---|
| `*IDN<n>?` | e.g. `Coherent, Inc - OBIS 488nm LX 30mW Laser - <serial> - <firmware> - <date>`. Through a controller, `*IDN?` names the controller (`OBIS MINI-CONTROLLER` on SWC's Remote). | **driver** |
| `SYST<n>:INF:MOD?` | Model text. Used to find which bays hold a laser (`ERR-…` = empty). | **driver** (Micro-Manager) |
| `SYST<n>:INF:SNUM?` | Serial number | **driver** |
| `SYST<n>:INF:WAV?` | Wavelength in nm, e.g. `489` (the actual wavelength, not the nominal one) | **driver** |
| `SYST<n>:INF:POW?` | Nominal (rated) power, W | **driver** (amsikking) |
| `SYST<n>:INF:TYP?` | `DDL` (diode) or `OPSL` (optically pumped). This decides the allowed modes. | **driver** (amsikking) |
| `SYST<n>:INF:FVER?` | Firmware version | **verify** |
| `SOUR<n>:POW:LIM:LOW?` | Lowest settable power, W. **Nonzero on OPSL** (an LS can't run at 0). | **driver** |
| `SOUR<n>:POW:LIM:HIGH?` | Highest settable power, W | **driver** |
| `SOUR<n>:POW:NOM?` | Nominal power, W | **driver** (storm-control) |

### Emission, mode and power

| Command | Reply | Source |
|---|---|---|
| `SOUR<n>:AM:STAT ON` / `OFF` | `OK`. Turns the laser on (Coherent Connection's Start) or off (Stop). | **driver** |
| `SOUR<n>:AM:STAT?` | `ON` / `OFF` | **driver** |
| `SOUR<n>:AM:INT CWP` / `CWC` | `OK`. Internal (CW) mode, constant power or constant current. | **driver** |
| `SOUR<n>:AM:EXT DIG` / `ANAL` / `MIX` / `DIGSO` / `MIXSO` | `OK`. External modulation (section 5). | **driver** |
| `SOUR<n>:AM:SOUR?` | The mode: `CWP`, `CWC`, `DIGITAL`, `ANALOG`, `MIXED`, `DIGSO`, `MIXSO` | **driver** |
| `SOUR<n>:POW:LEV:IMM:AMPL <W>` | `OK`. Sets the power setpoint. | **driver** |
| `SOUR<n>:POW:LEV:IMM:AMPL?` | The setpoint, W, e.g. `0.06937` | **driver** |
| `SOUR<n>:POW:LEV?` | The **measured** output power, W. It is 0 while a digital-mode laser's TTL is low. | **driver** |
| `SOUR<n>:POW:CURR?` | Diode current, A. Probably the mA figure Coherent Connection shows beside each laser. | **verify** |

### Health

| Command | Reply | Source |
|---|---|---|
| `SYST<n>:STAT?` | Status word, 8 hex digits (section 7) | **driver**, **seen** |
| `SYST<n>:FAUL?` | Fault word, 8 hex digits (section 7) | **driver** |
| `SOUR<n>:TEMP:BAS?` | Baseplate temperature, e.g. `24.5C` (note the unit suffix) | **driver** |
| `SOUR<n>:TEMP:DIOD?`, `SOUR<n>:TEMP:INT?` | Diode and internal temperatures | **driver** (Micro-Manager) |
| `SYST<n>:DIOD:HOUR?`, `SYST<n>:HOUR?` | Diode hours, system hours | **driver** (Micro-Manager), **verify** |
| `SYST<n>:ERR:COUN?`, `SYST<n>:ERR:NEXT?`, `SYST<n>:ERR:CLE` | The error queue: count, next, clear | **driver** (`CLE`), **verify** (rest) |
| `SYST<n>:LOCK?` | Interlock state | **verify** |

### Settings Coherent Connection exposes (photo 1)

| Coherent Connection | Command | Source |
|---|---|---|
| "Apply 5-second CDRH delay before starting emissions" | `SYST<n>:CDRH ON\|OFF` (and `?`) | **driver** |
| "Laser Auto Start" (emits at power-up, or when the key is turned) | `SYST<n>:AUT ON\|OFF` (and `?`) | **driver** |
| "Blanking Enabled" (analog input on LX lasers) | not known: find it in the manual | **verify** |
| "Power Cycle Laser" | not known | **verify** |
| "Field Recalibration" | not known. **Never send it from pyrpoc.** It recalibrates the laser's power. | — |

### Replies need parsing

Some replies carry units (`24.5C`). storm-control cut 6 characters off its `LIMit` replies, so some
firmware appends units to powers too (**driver**). Parse a number by stripping a known unit
suffix, and raise if what's left isn't a number. Don't guess.

When sending, write powers in fixed-point notation (`f"{watts:.5f}"`). Python's `str(1e-05)` gives
`1e-05`, which the laser may not accept (**verify**).

## 5. Operating modes

Coherent Connection names the modes differently from the commands. From the dropdown in photo 1
(**seen**) and the commands each driver sent (**driver**), the mapping is most likely:

| Coherent Connection | Set with | `AM:SOUR?` reads | What the light does |
|---|---|---|---|
| CW:Power | `AM:INT CWP` | `CWP` | Constant power at the setpoint while on. Closed loop on the internal photodiode. |
| CW:Current | `AM:INT CWC` | `CWC` | Constant diode current. Power drifts with temperature. |
| Digital:Current | `AM:EXT DIG` | `DIGITAL` | Setpoint while the DIGITAL input is high, off while it is low. No power feedback. |
| Digital:Power | `AM:EXT DIGSO` | `DIGSO` | As above, with power feedback. **DDL only.** |
| Analog:Power | `AM:EXT ANAL` | `ANALOG` | Output follows the ANALOG input voltage (0–5 V → 0–max, **verify**). The setpoint does not cap it. |
| Mixed:Current | `AM:EXT MIX` | `MIXED` | Digital gates, analog scales |
| Mixed:Power | `AM:EXT MIXSO` | `MIXSO` | As above, with power feedback. **DDL only.** |

The **Current/Power** pairing with `DIG`/`DIGSO` and `MIX`/`MIXSO` is inferred, not confirmed. To
confirm it, choose each mode in Coherent Connection, close it, and read `SOUR<n>:AM:SOUR?`
(section 11).

The OPSL lasers (4 and 5) offer fewer modes: Coherent Connection shows them only as "Digital
Modulation". amsikking sets `MIXSO` on DDL but `MIX` on OPSL, because OPSL has no `…SO` modes
(**driver**). The device should take the allowed modes from `INF:TYP?` and refuse the rest before
sending.

**Change mode with emission off.** amsikking always turns a laser off, changes the mode, then turns
it back on (**driver**). Whether the laser refuses a change while on (probably `ERR-221`, *Settings
conflict*) is **verify**.

amsikking also returns every laser to CWP on close, "to avoid 'checksum' error". The reason isn't
given. Record it as a known quirk, and check on the rig whether leaving a laser in an external mode
causes trouble at the next power-up.

## 6. Power

- **Setpoint vs nominal.** For the 488-30, Coherent Connection showed 5.0 mW as **16.7 %**, which is
  5/30. Its slider runs from 0 to **110 %** (**seen**). So the GUI's percent is of the nominal
  power (`INF:POW?`), and `POW:LIM:HIGH?` is probably 110 % of nominal (**verify**). The device
  should take its limits from `LIM:LOW?`/`LIM:HIGH?` and never assume 0 to nominal.
- **The setpoint is quantized.** Micro-Manager logs a warning when the echoed setpoint is far from
  the one requested (**driver**). Read the setpoint back after setting it, and report the value the
  laser actually took.
- **The setpoint does not limit analog modes.** In ANALOG and MIXED modes the input voltage sets
  the output, up to the laser's maximum (**driver**, SWC). A software power limit means nothing
  there.
- **Measured power is not the setpoint.** `POW:LEV?` is what the internal photodiode sees now. In
  digital mode it reads 0 while the TTL is low, which is the "0.0 mW Emitting" in the photos. It
  can take time to settle after a change: amsikking waits 1 s before reading it.

## 7. Status and fault words

`SYST<n>:STAT?` and `SYST<n>:FAUL?` each return 32 bits as 8 hex digits. Channel 0 gives the
controller's word. A laser's word appears to carry the controller's bits as well. The bits seen in
the photo-1 tooltip (**seen**):

```
Laser status       = 06000408
  00000008  Laser Standby
  00000400  Laser is Modulating
Controller status  = ce000488
  02000000  Controller: Key switch open
  04000000  Controller: interlock open
  08000000  Controller: Lasers are enumerated
  40000000  Controller: Remote is active
```

The laser's `06000408` carries the controller's key-switch (`02…`) and interlock (`04…`) bits. The
controller's `ce000488` also has `80000000` and `00000080` set, which the tooltip didn't label (it
was cut off). SWC saw `FAUL?` = `80000000` with no fault anywhere (**driver**). So **don't treat a
nonzero word as a fault.** Decode the named bits, and log the raw hex.

The other bits, as recalled from the manual, are below. All are **verify**: copy the real tables
from the operator's manual into this section.

| Bit | Status meaning (recalled) |
|---|---|
| `00000001` | Laser fault |
| `00000002` | Laser emission |
| `00000004` | Laser ready |
| `00000008` | Laser standby (**seen**) |
| `00000010` | CDRH delay in progress |
| `00000020` | Laser hardware fault |
| `00000040` | Laser error |
| `00000080` | Laser power calibration |
| `00000100` | Laser warm up |
| `00000200` | Laser noise |
| `00000400` | External operating mode / modulating (**seen**) |
| `00000800` | Field calibration |
| `02000000` | Controller: key switch open / standby (**seen**) |
| `04000000` | Controller: interlock open (**seen**) |
| `08000000` | Controller: lasers enumerated (**seen**) |
| `10000000` | Controller error |
| `20000000` | Controller fault |
| `40000000` | Controller: remote active (**seen**) |
| `80000000` | Controller indicator (?) |

The fault word has its own table (temperatures, power supply, I2C, over-current, checksum, warm-up
limit, …). The bit positions aren't recalled reliably enough to list here. Get them from the manual.

**An open question from the photo.** The tooltip says *interlock open* although the interlock jumper
is installed. The photo was taken with the key at STANDBY. Check whether `04000000` clears with the
key ON. If it doesn't, either the label is misleading or the jumper isn't making contact.

## 8. Emission and safety

Light leaves a laser only when **all** of the following hold. The hardware enforces the first two,
and software can't override them:

1. **The key switch is at ON.** At STANDBY, every laser shows *Key Switch Standby* (**seen**).
2. **The interlock is closed** (the jumper on the green terminal).
3. **The laser is enabled** (`SOUR<n>:AM:STAT ON`). If CDRH is ON, emission starts 5 s later
   (**driver**: amsikking refuses to enable with CDRH on).
4. **The laser has warmed up and is ready.** This is quick for the diodes. The OPSL lasers may take
   longer (**verify**: watch the ready and warm-up bits).
5. **In a digital or mixed mode, the DIGITAL input is high.** In an analog or mixed mode, the
   ANALOG voltage also sets the level.

What follows for the device:

- **Don't write settings the user may have tuned elsewhere,** such as CDRH, auto-start and mode,
  as a side effect of connecting. The Prior stage follows the same rule. Read them at connect and
  show them. Change them only when asked.
- **Know what CDRH costs.** With CDRH ON, `AM:STAT ON` returns at once, but light arrives 5 s later.
  The device must either wait for the ready/emission bit or refuse with a clear message. Decide
  which with the user.
- **Know what auto-start does.** With auto-start ON, a laser emits as soon as the box powers up or
  the key is turned, with no software involved. Show it plainly.
- **Disconnecting.** Today the rig leaves lasers enabled in digital mode, gated by TTL (**seen**).
  If pyrpoc turns every laser off on disconnect, that is safe, but it changes how the rig is used.
  If it leaves them on, a crashed pyrpoc leaves a laser that emits whenever its TTL is high. **Ask
  the user which.** Either way, if Python is killed the laser keeps its last state. Only the key,
  the interlock and the TTL source stop it then.
- **Use one lock per port.** The box is one serial line shared by six addressees. A panel polling
  status while a program sets power must not interleave writes and reads. Serialize each full
  exchange (write, then read to `OK`/`ERR`) under a lock, as the Prior SDK wrapper does.

## 9. Fast gating goes through the SMB inputs, not the serial port

Each serial exchange is a USB round trip plus the laser's processing: milliseconds at best. That is
fine for setpoints, modes and status, but far too slow and jittery for gating light to pixels,
lines or frames. The rig already gates through the **DIGITAL SMB inputs** (**seen**), presumably
from the DAQ's digital lines, which pyrpoc already drives (`mask_ttl`, `daq_tasks`). So keep two
paths:

- **Serial (this device):** identity, limits, mode, setpoint, enable, status. Changed between runs,
  or between passes at most.
- **TTL (existing DAQ path):** when the light is on, inside a run.

To record on the rig (**verify**, from the Laser Box datasheet):

- The digital input's thresholds and input impedance. The box comes in **2 kΩ and 50 Ω** input
  versions (amsikking tested both). A 50 Ω input may load a DAQ line more than it can drive.
- The analog input's range.
- The modulation bandwidth for LX vs LS. LS (OPSL) modulates far slower than LX.

## 10. What this means for the pyrpoc device

These are hardware constraints for the integration session, not a design:

- **One connection per box.** All five lasers share `COM8`, so there is one serial session, opened
  and closed as a whole (`Connection.SESSION`, like the Prior stage). Whether the lasers then appear
  as one device or five is an integration decision. Either way, all five share one session and one
  lock.
- **Find the bays at connect.** Query `SYST<n>:INF:MOD?` for n = 1…5. `ERR-241` means the bay is
  empty. Then read each laser's identity, type and limits. Don't hard-code the five lasers in the
  photos.
- **Keep the layers apart,** as `prior_stage/sdk.py` does:
  - a **transport**: open, write a line, read a line, close;
  - a **protocol**: framing to `OK`/`ERR`, error codes, number parsing, the channel prefix;
  - **laser operations** in watts and the box's own mode names.
  Then the device holds only what pyrpoc needs.
- **Test with a simulated transport.** Without hardware, everything above the transport can run
  against a fake box. It answers the commands in section 4 with the framing in section 3, keeps the
  state those commands change, and can be told to answer `ERR-…` or to go silent. SWC's MATLAB
  driver did exactly this (71 tests before first contact). Build it from this document, and
  correct both together after the rig checks.
- **Add `pyserial` as a dependency** in `pyproject.toml`. Open the port as `COM<n>` from an
  `int_field`, like the other serial devices, with a read timeout (2 s is plenty; amsikking used 5).
  A timeout is a fault, not an empty reply.

## 11. Rig checks

Do these on the rig PC before, or alongside, writing the driver. Block the beams for any check that
turns a laser on.

**First, read-only.** Close Coherent Connection, then run the probe beside this document. It sends
only queries, never a setting, and prints every raw reply:

```
pip install pyserial
python docs/obis_probe.py COM8 > obis_probe_COM8.txt
```

Commit the transcript beside this document. It settles most **verify** marks at once:

- the framing (`OK` after a value? any prompt? `\r\n`?);
- whether short and long forms both work with a channel number;
- what an unnumbered command reaches;
- each bay's model, type, limits, mode and setpoint;
- the status and fault words.

**Then with Coherent Connection.** These need the GUI open, so the probe can't run at the same time:

1. Put the key at ON. Does `04000000` (*interlock open*) clear from the controller status?
2. Choose each mode in turn for one DDL laser, close Coherent Connection, run the probe, and record
   what `AM:SOUR?` reads. This confirms the mapping in section 5.
3. Open the operator's manual (from the USB stick shipped with the lasers, or from Coherent's site)
   and copy into section 7 the status and fault bit tables, the commands for blanking and power
   cycling, and the error queue commands.

**Then with commands, beams blocked:**

1. Turn handshaking on (`SYST<n>:COMM:HAND ON`) and confirm every exchange ends in `OK` or `ERR`.
2. Set a setpoint and read it back. How large is the quantization?
3. Send a setpoint below `LIM:LOW` and one above `LIM:HIGH`. Is it refused (which `ERR`) or
   clamped?
4. Change a laser's mode while it is ON. Is it refused (`ERR-221`?) or applied?
5. Run `AM:STAT ON` with CDRH ON. Which status bits show the delay, and when does emission start?
6. Time a round trip, such as a `SYST1:STAT?` exchange, to know what polling costs.
7. Measure one TTL-gated pulse on a photodiode or power meter to confirm gating and its delay.

## Sources

- [amsikking/coherent_OBIS_LSLX_laser_box][amsikking]: Python adaptor tested on the OBIS LX/LS Laser
  Box (SKU 1343229, 2 kΩ and 50 Ω versions). Channel addressing, framing, error codes, `INF:TYP?`,
  OPSL handling.
- [Micro-Manager CoherentOBIS and CoherentScientificRemote][mm]: C++ adapters. Short forms with a
  channel number, handshake/prompt at connect, power in W.
- [SainsburyWellcomeCentre/OBISLaser][swc]: MATLAB driver whose `docs/scpi.md` records real replies
  from an OBIS behind a Remote. Short-form quirk, `FAUL?` = `80000000`,
  `CoherentConnectionLibrary.dll`.
- [storm-control `obis.py`][storm]: early Python driver. Unit suffixes on replies.
- Coherent *OBIS LX/LS Operator's Manual* (P/N 2283534; older revision 1184163), *OBIS LX/LS Laser
  Box* datasheet, Coherent Connection user guide. These are the authoritative references, and they
  could not be fetched when this was written.

[amsikking]: https://github.com/amsikking/coherent_OBIS_LSLX_laser_box
[mm]: https://github.com/micro-manager/mmCoreAndDevices/tree/main/DeviceAdapters/CoherentOBIS
[swc]: https://github.com/SainsburyWellcomeCentre/OBISLaser
[storm]: https://github.com/AlistairBoettiger/storm-control/blob/master/sc_hardware/coherent/obis.py
[pyserial]: https://pypi.org/project/pyserial/
