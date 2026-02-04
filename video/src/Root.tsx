import { Composition } from "remotion";
import { TribalMemoryDemo } from "./TribalMemoryDemo";

export const RemotionRoot: React.FC = () => {
  return (
    <>
      <Composition
        id="TribalMemoryDemo"
        component={TribalMemoryDemo}
        durationInFrames={1230}
        fps={30}
        width={1920}
        height={1080}
      />
    </>
  );
};
