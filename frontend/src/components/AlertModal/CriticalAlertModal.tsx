// ============================================================
// Critical Alert Modal - 긴급 알람 풀스크린 모달
// ============================================================
import React, { useEffect, useRef, useState } from 'react';
import { alertSound } from '@/utils/alertSound';
import type { Alarm } from '@/types';

interface CriticalAlertModalProps {
  alarm: Alarm;
  onAcknowledge: (alarmId: string, note?: string) => void;
  onEscalate?: (alarmId: string) => void;
}

export const CriticalAlertModal: React.FC<CriticalAlertModalProps> = ({
  alarm,
  onAcknowledge,
  onEscalate,
}) => {
  const [note, setNote] = useState('');
  const [isAcknowledging, setIsAcknowledging] = useState(false);
  const soundIntervalRef = useRef<NodeJS.Timeout | null>(null);

  // 반복 알람 소리
  useEffect(() => {
    // 초기 소리
    alertSound.playCritical();

    // 3초마다 반복
    soundIntervalRef.current = setInterval(() => {
      alertSound.playCritical();
    }, 3000);

    return () => {
      if (soundIntervalRef.current) {
        clearInterval(soundIntervalRef.current);
      }
    };
  }, []);

  const handleAcknowledge = () => {
    setIsAcknowledging(true);

    // 소리 중지
    if (soundIntervalRef.current) {
      clearInterval(soundIntervalRef.current);
    }

    onAcknowledge(alarm.alarm_id, note);
  };

  const handleEscalate = () => {
    if (soundIntervalRef.current) {
      clearInterval(soundIntervalRef.current);
    }
    onEscalate?.(alarm.alarm_id);
  };

  const formatTime = (timestamp: string) => {
    const date = new Date(timestamp);
    return date.toLocaleTimeString('ko-KR', {
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
      hour12: false
    });
  };

  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center">
      {/* Backdrop with pulse animation */}
      <div className="absolute inset-0 bg-red-900/90 animate-pulse-slow" />

      {/* Alert content */}
      <div className="relative z-10 w-full max-w-2xl mx-4">
        {/* Warning icon */}
        <div className="flex justify-center mb-6">
          <div className="w-24 h-24 rounded-full bg-red-500 flex items-center justify-center animate-bounce-slow">
            <svg
              className="w-16 h-16 text-white"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
              />
            </svg>
          </div>
        </div>

        {/* Title */}
        <h1 className="text-4xl font-bold text-white text-center mb-2 animate-pulse">
          CRITICAL ALERT
        </h1>
        <p className="text-red-200 text-center text-lg mb-8">
          즉각적인 조치가 필요합니다
        </p>

        {/* Alert details card */}
        <div className="bg-slate-800/90 rounded-xl p-6 mb-6 border-2 border-red-500">
          <div className="grid grid-cols-2 gap-4 mb-4">
            <div>
              <p className="text-sm text-slate-400">Equipment</p>
              <p className="text-xl font-bold text-white">{alarm.equipment_id}</p>
            </div>
            <div>
              <p className="text-sm text-slate-400">Time</p>
              <p className="text-xl font-bold text-white">{formatTime(alarm.occurred_at)}</p>
            </div>
          </div>

          <div className="mb-4">
            <p className="text-sm text-slate-400">Alarm Code</p>
            <p className="text-lg font-mono text-red-400">{alarm.alarm_code || 'N/A'}</p>
          </div>

          <div className="bg-slate-900/50 rounded-lg p-4">
            <p className="text-sm text-slate-400 mb-1">Message</p>
            <p className="text-white text-lg">{alarm.message || 'Critical condition detected'}</p>
          </div>
        </div>

        {/* Note input */}
        <div className="mb-6">
          <label className="block text-sm text-slate-300 mb-2">
            조치 내용 (선택사항)
          </label>
          <textarea
            value={note}
            onChange={(e) => setNote(e.target.value)}
            placeholder="확인 사항이나 조치 내용을 입력하세요..."
            className="w-full px-4 py-3 bg-slate-800 border border-slate-600 rounded-lg text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-red-500"
            rows={2}
          />
        </div>

        {/* Action buttons */}
        <div className="flex gap-4">
          <button
            onClick={handleAcknowledge}
            disabled={isAcknowledging}
            className="flex-1 py-4 px-6 bg-red-600 hover:bg-red-500 disabled:bg-red-800 text-white font-bold text-lg rounded-lg transition-colors flex items-center justify-center gap-2"
          >
            {isAcknowledging ? (
              <>
                <svg className="w-5 h-5 animate-spin" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                </svg>
                처리 중...
              </>
            ) : (
              <>
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                </svg>
                확인 (Acknowledge)
              </>
            )}
          </button>

          {onEscalate && (
            <button
              onClick={handleEscalate}
              className="py-4 px-6 bg-amber-600 hover:bg-amber-500 text-white font-bold rounded-lg transition-colors flex items-center gap-2"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 10l7-7m0 0l7 7m-7-7v18" />
              </svg>
              에스컬레이션
            </button>
          )}
        </div>

        {/* Instructions */}
        <p className="text-center text-slate-400 text-sm mt-4">
          이 알람은 확인 버튼을 누를 때까지 계속 표시됩니다
        </p>
      </div>

      {/* Corner flashing indicators */}
      <div className="absolute top-4 left-4 w-4 h-4 rounded-full bg-red-500 animate-ping" />
      <div className="absolute top-4 right-4 w-4 h-4 rounded-full bg-red-500 animate-ping" />
      <div className="absolute bottom-4 left-4 w-4 h-4 rounded-full bg-red-500 animate-ping" />
      <div className="absolute bottom-4 right-4 w-4 h-4 rounded-full bg-red-500 animate-ping" />
    </div>
  );
};
