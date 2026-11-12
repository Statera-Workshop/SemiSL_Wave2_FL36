#!/usr/bin/env bash
# queue_runner.sh - one GPU job at a time (SSL Wave 1a/1b, Ubuntu RTX 4070).
# Jobs: ~/SSL_Wave1/queue/jobs.txt  "name | est minutes | command" (appendable while running).
# Skip if now+est > CUTOFF; at HARD_STOP the running job gets SIGTERM, SIGKILL 60 s later.
# Stop now: touch ~/SSL_Wave1/queue/STOP. State: queue/state/<name>.start/.done/.fail/.skip/.stopped
set -u
BASE="${SSL_BASE:-$HOME/SSL_Wave1}"
Q="$BASE/queue"; L="$BASE/logs"
CUTOFF="${CUTOFF:-2026-11-09 06:00}"
HARD_STOP="${HARD_STOP:-2026-11-09 06:10}"
mkdir -p "$Q/state" "$L"
touch "$Q/jobs.txt"
export PHYSICAL_AI_ROOT="$BASE" PYTHONUNBUFFERED=1
cut_s=$(date -d "$CUTOFF" +%s)
hard_s=$(date -d "$HARD_STOP" +%s)
log() { echo "$(date '+%F %T')  $*" | tee -a "$L/queue.log"; }
trim() { local s="$1"; s="${s#"${s%%[![:space:]]*}"}"; s="${s%"${s##*[![:space:]]}"}"; printf '%s' "$s"; }
log "queue start: cutoff $CUTOFF, hard stop $HARD_STOP, python $(command -v python)"
while true; do
  if [ -f "$Q/STOP" ]; then log "STOP file - queue ends"; break; fi
  now=$(date +%s)
  if [ "$now" -ge "$cut_s" ]; then log "cutoff reached - queue ends"; break; fi
  next=""; nest=0; ncmd=""
  while IFS='|' read -r name est cmd || [ -n "$name" ]; do
    name=$(trim "$name")
    [ -z "$name" ] && continue
    case "$name" in \#*) continue ;; esac
    for s in done fail skip stopped; do [ -e "$Q/state/$name.$s" ] && continue 2; done
    next="$name"; nest=$(trim "$est"); ncmd=$(trim "$cmd"); break
  done < "$Q/jobs.txt"
  if [ -z "$next" ]; then sleep 60; continue; fi
  if ! [[ "$nest" =~ ^[0-9]+$ ]]; then log "FAIL $next: estimate '$nest' is not a number"; touch "$Q/state/$next.fail"; continue; fi
  if [ $(( now + nest * 60 )) -gt "$cut_s" ]; then
    log "SKIP $next: needs ~${nest} min, would end after the cutoff"; touch "$Q/state/$next.skip"; continue
  fi
  log "START $next (~${nest} min): $ncmd"
  date '+%F %T' > "$Q/state/$next.start"
  setsid bash -c "$ncmd" > "$L/job_$next.out" 2>&1 &
  pid=$!
  interrupted=0
  while kill -0 "$pid" 2>/dev/null; do
    sleep 20
    if [ "$interrupted" -eq 0 ] && { [ -f "$Q/STOP" ] || [ "$(date +%s)" -ge "$hard_s" ]; }; then
      log "INTERRUPT $next (STOP file or hard stop): SIGTERM"
      kill -TERM -- "-$pid" 2>/dev/null
      interrupted=1; t_int=$(date +%s)
    fi
    if [ "$interrupted" -eq 1 ] && [ $(( $(date +%s) - t_int )) -ge 60 ] && kill -0 "$pid" 2>/dev/null; then
      log "INTERRUPT $next: still running after 60 s, SIGKILL"
      kill -KILL -- "-$pid" 2>/dev/null
      interrupted=2
    fi
  done
  wait "$pid"; rc=$?
  mins=$(( ( $(date +%s) - now ) / 60 ))
  if [ "$interrupted" -ne 0 ]; then
    log "STOPPED $next after ${mins} min (rc $rc) - resumable, delete state/$next.stopped to run it again"
    touch "$Q/state/$next.stopped"
  elif [ "$rc" -eq 0 ]; then
    log "DONE $next in ${mins} min"; touch "$Q/state/$next.done"
  else
    log "FAIL $next rc $rc after ${mins} min - see logs/job_$next.out"; touch "$Q/state/$next.fail"
  fi
done
log "queue end"
