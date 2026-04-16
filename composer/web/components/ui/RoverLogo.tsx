interface RoverLogoProps {
  size?: number;
  className?: string;
}

export function RoverLogo({ size = 80, className }: RoverLogoProps) {
  const s = size;
  const cx = s / 2;

  return (
    <svg
      width={s}
      height={s}
      viewBox="0 0 80 80"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className={className}
    >
      {/* Antenna */}
      <line x1="40" y1="6" x2="40" y2="14" stroke="#fb923c" strokeWidth="2" strokeLinecap="round" />
      <circle cx="40" cy="5" r="2.5" fill="#fb923c" />

      {/* Head — binocular box */}
      <rect x="18" y="14" width="44" height="28" rx="5" fill="#fb923c" />
      <rect x="18" y="14" width="44" height="28" rx="5" stroke="#ea580c" strokeWidth="1" />

      {/* Left eye tube */}
      <rect x="22" y="19" width="16" height="16" rx="8" fill="#1c1c2e" />
      {/* Left iris */}
      <circle cx="30" cy="27" r="6" fill="#ea580c" />
      {/* Left pupil */}
      <circle cx="30" cy="27" r="3" fill="#1c1c2e" />
      {/* Left highlight */}
      <circle cx="32" cy="25" r="1.2" fill="white" opacity="0.9" />

      {/* Right eye tube */}
      <rect x="42" y="19" width="16" height="16" rx="8" fill="#1c1c2e" />
      {/* Right iris */}
      <circle cx="50" cy="27" r="6" fill="#ea580c" />
      {/* Right pupil */}
      <circle cx="50" cy="27" r="3" fill="#1c1c2e" />
      {/* Right highlight */}
      <circle cx="52" cy="25" r="1.2" fill="white" opacity="0.9" />

      {/* Body */}
      <rect x="27" y="42" width="26" height="20" rx="4" fill="#fb923c" />
      <rect x="27" y="42" width="26" height="20" rx="4" stroke="#ea580c" strokeWidth="1" />
      {/* Vent slots */}
      <rect x="31" y="46" width="18" height="2" rx="1" fill="#ea580c" opacity="0.6" />
      <rect x="31" y="51" width="18" height="2" rx="1" fill="#ea580c" opacity="0.6" />
      <rect x="31" y="56" width="18" height="2" rx="1" fill="#ea580c" opacity="0.6" />

      {/* Left arm — ball gripper */}
      <line x1="27" y1="48" x2="18" y2="52" stroke="#fb923c" strokeWidth="3" strokeLinecap="round" />
      <circle cx="16" cy="53" r="4" fill="#fb923c" stroke="#ea580c" strokeWidth="1" />
      <circle cx="16" cy="53" r="1.5" fill="#ea580c" opacity="0.5" />

      {/* Right arm — ball gripper */}
      <line x1="53" y1="48" x2="62" y2="52" stroke="#fb923c" strokeWidth="3" strokeLinecap="round" />
      <circle cx="64" cy="53" r="4" fill="#fb923c" stroke="#ea580c" strokeWidth="1" />
      <circle cx="64" cy="53" r="1.5" fill="#ea580c" opacity="0.5" />

      {/* Left tank leg */}
      <rect x="22" y="62" width="14" height="12" rx="3" fill="#ea580c" />
      {/* Left traction ridges */}
      <line x1="25" y1="62" x2="25" y2="74" stroke="#fb923c" strokeWidth="1" opacity="0.5" />
      <line x1="28" y1="62" x2="28" y2="74" stroke="#fb923c" strokeWidth="1" opacity="0.5" />
      <line x1="31" y1="62" x2="31" y2="74" stroke="#fb923c" strokeWidth="1" opacity="0.5" />
      {/* Left wheels */}
      <circle cx="24" cy="74" r="4" fill="#1c1c2e" stroke="#fb923c" strokeWidth="1.5" />
      <circle cx="24" cy="74" r="1.5" fill="#fb923c" />
      <circle cx="34" cy="74" r="4" fill="#1c1c2e" stroke="#fb923c" strokeWidth="1.5" />
      <circle cx="34" cy="74" r="1.5" fill="#fb923c" />

      {/* Right tank leg */}
      <rect x="44" y="62" width="14" height="12" rx="3" fill="#ea580c" />
      {/* Right traction ridges */}
      <line x1="47" y1="62" x2="47" y2="74" stroke="#fb923c" strokeWidth="1" opacity="0.5" />
      <line x1="50" y1="62" x2="50" y2="74" stroke="#fb923c" strokeWidth="1" opacity="0.5" />
      <line x1="53" y1="62" x2="53" y2="74" stroke="#fb923c" strokeWidth="1" opacity="0.5" />
      {/* Right wheels */}
      <circle cx="46" cy="74" r="4" fill="#1c1c2e" stroke="#fb923c" strokeWidth="1.5" />
      <circle cx="46" cy="74" r="1.5" fill="#fb923c" />
      <circle cx="56" cy="74" r="4" fill="#1c1c2e" stroke="#fb923c" strokeWidth="1.5" />
      <circle cx="56" cy="74" r="1.5" fill="#fb923c" />
    </svg>
  );
}
