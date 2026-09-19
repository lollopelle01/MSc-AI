#ifndef _STATS_H
#define _STATS_H

#ifdef BOARD
// INSERT BOARD-SPECIFIC PROFILING IF NEEDED
#else

#ifdef STATS

/* Global variables: define once at file scope.
   Using `static` to limit la visibilità al file di compilazione.
*/
static unsigned long _cycles = 0;
static unsigned long _instr = 0;
static unsigned long _active = 0;
static unsigned long _ldext = 0;
static unsigned long _tcdmcont = 0;
static unsigned long _ldstall = 0;
static unsigned long _imiss = 0;

/* INIT_STATS: azzera i contatori (usare all'inizio, non per dichiarare variabili) */
#define INIT_STATS() \
    do { \
      _cycles = 0; \
      _instr = 0; \
      _active = 0; \
      _ldext = 0; \
      _tcdmcont = 0; \
      _ldstall = 0; \
      _imiss = 0; \
    } while(0)

/* RESET_STATS: alias per azzerare i contatori durante l'esecuzione */
#define RESET_STATS() INIT_STATS()

/* PRE_START_STATS: configura i contatori hardware (se richiesto dalla piattaforma) */
#define PRE_START_STATS()  \
    pi_perf_conf((1<<PI_PERF_CYCLES) | (1<<PI_PERF_INSTR) | (1<<PI_PERF_ACTIVE_CYCLES) | (1<<PI_PERF_LD_EXT) | (1<<PI_PERF_TCDM_CONT) | (1<<PI_PERF_LD_STALL) | (1<<PI_PERF_IMISS) );

/* START/STOP: avviano e fermano il conteggio hardware.
   STOP_STATS legge i valori e stampa SOLO se eseguito dal core 0.
*/
#define START_STATS()  \
    do { \
      pi_perf_stop(); \
      pi_perf_reset(); \
      pi_perf_start(); \
    } while(0)

#define STOP_STATS() \
    do { \
      pi_perf_stop(); \
      _cycles   += pi_perf_read (PI_PERF_CYCLES); \
      _instr    += pi_perf_read (PI_PERF_INSTR); \
      _active   += pi_perf_read (PI_PERF_ACTIVE_CYCLES); \
      _ldext    += pi_perf_read (PI_PERF_LD_EXT); \
      _tcdmcont += pi_perf_read (PI_PERF_TCDM_CONT); \
      _ldstall  += pi_perf_read (PI_PERF_LD_STALL); \
      _imiss    += pi_perf_read (PI_PERF_IMISS); \
      if (pi_core_id() == 0) { \
        printf("\n[core 0] cycles = %lu\n", _cycles); \
        printf("[core 0] instr = %lu\n", _instr); \
        printf("[core 0] active cycles = %lu\n", _active); \
        printf("[core 0] ext load = %lu\n", _ldext); \
        printf("[core 0] TCDM cont = %lu\n", _tcdmcont); \
        printf("[core 0] ld stall = %lu\n", _ldstall); \
        printf("[core 0] imiss = %lu\n", _imiss); \
      } \
    } while(0)

#else  /* STATS disabled */

#define INIT_STATS()
#define RESET_STATS()
#define PRE_START_STATS()
#define START_STATS()
#define STOP_STATS()

#endif /* STATS */

#endif /* BOARD */

#endif /* _STATS_H */
