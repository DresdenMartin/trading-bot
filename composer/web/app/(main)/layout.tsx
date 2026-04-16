import { MainLayoutClient } from "@/components/layout/main-layout-client";

export default function MainLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <div className="flex h-full bg-zinc-950">
      <MainLayoutClient>{children}</MainLayoutClient>
    </div>
  );
}
