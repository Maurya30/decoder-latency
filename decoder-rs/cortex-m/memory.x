/* Where the linker may put things.

   These addresses are the memory of QEMU's "mps2-an386" machine (ARM's MPS2 development
   board running its Cortex-M4 image), read from QEMU itself with `info mtree`. It is an
   emulator's layout, used only so the image can be run. It is not a chosen target board.

   FLASH holds the code and the model's weights, and is only ever read.
   RAM holds everything that changes while it runs. The emulated machine has 4 MB there;
   the image is given just the first 64 kB, to show that it fits. */
MEMORY
{
  FLASH : ORIGIN = 0x00000000, LENGTH = 4M
  RAM   : ORIGIN = 0x20000000, LENGTH = 64K
}
