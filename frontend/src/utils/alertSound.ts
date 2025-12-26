// ============================================================
// Alert Sound Utility - Web Audio API 기반 알람 소리
// ============================================================

type SoundType = 'critical' | 'warning' | 'info';

class AlertSoundManager {
  private audioContext: AudioContext | null = null;
  private enabled: boolean = true;
  private volume: number = 0.5;

  private getAudioContext(): AudioContext {
    if (!this.audioContext) {
      this.audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
    }
    return this.audioContext;
  }

  /**
   * 비프음 생성
   */
  private playBeep(frequency: number, duration: number, type: OscillatorType = 'sine'): void {
    if (!this.enabled) return;

    try {
      const ctx = this.getAudioContext();
      const oscillator = ctx.createOscillator();
      const gainNode = ctx.createGain();

      oscillator.connect(gainNode);
      gainNode.connect(ctx.destination);

      oscillator.frequency.value = frequency;
      oscillator.type = type;

      gainNode.gain.setValueAtTime(this.volume, ctx.currentTime);
      gainNode.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + duration);

      oscillator.start(ctx.currentTime);
      oscillator.stop(ctx.currentTime + duration);
    } catch (e) {
      console.warn('Alert sound failed:', e);
    }
  }

  /**
   * Critical 알람 - 긴급 경고음 (높은 톤, 반복)
   */
  playCritical(): void {
    // 3회 반복 비프
    this.playBeep(880, 0.15, 'square');
    setTimeout(() => this.playBeep(880, 0.15, 'square'), 200);
    setTimeout(() => this.playBeep(880, 0.15, 'square'), 400);
  }

  /**
   * Warning 알람 - 주의 경고음 (중간 톤)
   */
  playWarning(): void {
    // 2회 반복 비프
    this.playBeep(660, 0.2, 'triangle');
    setTimeout(() => this.playBeep(660, 0.2, 'triangle'), 300);
  }

  /**
   * Info 알람 - 정보 알림음 (낮은 톤)
   */
  playInfo(): void {
    // 1회 부드러운 비프
    this.playBeep(440, 0.3, 'sine');
  }

  /**
   * 알람 유형에 따른 소리 재생
   */
  play(type: SoundType): void {
    switch (type) {
      case 'critical':
        this.playCritical();
        break;
      case 'warning':
        this.playWarning();
        break;
      case 'info':
        this.playInfo();
        break;
    }
  }

  /**
   * 연결 성공음
   */
  playConnected(): void {
    this.playBeep(523, 0.1, 'sine'); // C5
    setTimeout(() => this.playBeep(659, 0.1, 'sine'), 100); // E5
    setTimeout(() => this.playBeep(784, 0.15, 'sine'), 200); // G5
  }

  /**
   * 연결 끊김 알림음
   */
  playDisconnected(): void {
    this.playBeep(784, 0.1, 'sine'); // G5
    setTimeout(() => this.playBeep(659, 0.1, 'sine'), 100); // E5
    setTimeout(() => this.playBeep(523, 0.2, 'sine'), 200); // C5
  }

  /**
   * 테스트 사운드
   */
  playTest(): void {
    this.playBeep(440, 0.5, 'sine');
  }

  /**
   * 소리 활성화/비활성화
   */
  setEnabled(enabled: boolean): void {
    this.enabled = enabled;
  }

  isEnabled(): boolean {
    return this.enabled;
  }

  /**
   * 볼륨 설정 (0.0 ~ 1.0)
   */
  setVolume(volume: number): void {
    this.volume = Math.max(0, Math.min(1, volume));
  }

  getVolume(): number {
    return this.volume;
  }

  /**
   * AudioContext 초기화 (사용자 인터랙션 필요)
   */
  async initialize(): Promise<boolean> {
    try {
      const ctx = this.getAudioContext();
      if (ctx.state === 'suspended') {
        await ctx.resume();
      }
      return true;
    } catch (e) {
      console.error('Failed to initialize audio:', e);
      return false;
    }
  }
}

export const alertSound = new AlertSoundManager();
