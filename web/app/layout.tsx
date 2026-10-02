import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "ROC-Calib · Camera–LiDAR Calibration",
  description: "从ROS2 bag中选择静止观测，同步显示点云、原始图像和当前参数投影。",
  icons: { icon: "/favicon.svg" },
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="zh-CN">
      <body>{children}</body>
    </html>
  );
}
