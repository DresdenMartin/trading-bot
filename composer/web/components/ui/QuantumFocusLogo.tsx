interface QuantumFocusLogoProps {
  size?: number;
  className?: string;
}

export function QuantumFocusLogo({ size = 80, className }: QuantumFocusLogoProps) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 80 80"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className={className}
    >
      {/* Background circle */}
      <circle cx="40" cy="40" r="38" fill="#1a1840" />

      {/* Outer diamond frame */}
      <polygon
        points="40,4 76,40 40,76 4,40"
        stroke="#4F46E5"
        strokeWidth="1.5"
        fill="none"
        opacity="0.6"
      />

      {/* Cardinal tick marks */}
      <line x1="40" y1="4" x2="40" y2="10" stroke="#818cf8" strokeWidth="1.5" strokeLinecap="round" />
      <line x1="40" y1="70" x2="40" y2="76" stroke="#818cf8" strokeWidth="1.5" strokeLinecap="round" />
      <line x1="4" y1="40" x2="10" y2="40" stroke="#818cf8" strokeWidth="1.5" strokeLinecap="round" />
      <line x1="70" y1="40" x2="76" y2="40" stroke="#818cf8" strokeWidth="1.5" strokeLinecap="round" />

      {/* Diagonal tick dots */}
      <circle cx="17" cy="17" r="1.5" fill="#4F46E5" opacity="0.5" />
      <circle cx="63" cy="17" r="1.5" fill="#4F46E5" opacity="0.5" />
      <circle cx="17" cy="63" r="1.5" fill="#4F46E5" opacity="0.5" />
      <circle cx="63" cy="63" r="1.5" fill="#4F46E5" opacity="0.5" />

      {/* Almond eye shape (clip path) */}
      <defs>
        <clipPath id="eyeClip">
          <ellipse cx="40" cy="40" rx="26" ry="14" />
        </clipPath>
      </defs>

      {/* Eye white */}
      <ellipse cx="40" cy="40" rx="26" ry="14" fill="#0f0e2a" />

      {/* Outer iris ring */}
      <circle cx="40" cy="40" r="13" fill="#1a1840" stroke="#4F46E5" strokeWidth="1" />

      {/* Concentric iris rings */}
      <circle cx="40" cy="40" r="11" stroke="#818cf8" strokeWidth="0.75" fill="none" opacity="0.5" />
      <circle cx="40" cy="40" r="8" stroke="#4F46E5" strokeWidth="0.75" fill="none" opacity="0.7" />

      {/* Radial spokes */}
      {Array.from({ length: 12 }).map((_, i) => {
        const angle = (i * 30 * Math.PI) / 180;
        const x1 = 40 + 5 * Math.cos(angle);
        const y1 = 40 + 5 * Math.sin(angle);
        const x2 = 40 + 11 * Math.cos(angle);
        const y2 = 40 + 11 * Math.sin(angle);
        return (
          <line
            key={i}
            x1={x1}
            y1={y1}
            x2={x2}
            y2={y2}
            stroke="#818cf8"
            strokeWidth="0.6"
            opacity="0.5"
          />
        );
      })}

      {/* Pupil */}
      <circle cx="40" cy="40" r="5" fill="#4F46E5" />
      <circle cx="40" cy="40" r="3" fill="#1a0050" />

      {/* Diamond catchlight */}
      <polygon points="40,36 42,40 40,44 38,40" fill="white" opacity="0.85" />

      {/* Eye outline */}
      <ellipse
        cx="40"
        cy="40"
        rx="26"
        ry="14"
        stroke="#818cf8"
        strokeWidth="1.5"
        fill="none"
      />

      {/* Eye corner points */}
      <circle cx="14" cy="40" r="1.5" fill="#818cf8" opacity="0.8" />
      <circle cx="66" cy="40" r="1.5" fill="#818cf8" opacity="0.8" />
    </svg>
  );
}
