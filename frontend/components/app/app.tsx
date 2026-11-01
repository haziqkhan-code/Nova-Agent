'use client';

import { useEffect, useMemo } from 'react';
import { TokenSource } from 'livekit-client';
import { useRoomContext, useSession } from '@livekit/components-react';
import { WarningIcon } from '@phosphor-icons/react/dist/ssr';
import { toast } from 'sonner';
import type { AppConfig } from '@/app-config';
import { AgentSessionProvider } from '@/components/agents-ui/agent-session-provider';
import { StartAudioButton } from '@/components/agents-ui/start-audio-button';
import { JarvisBackground } from '@/components/app/jarvis-background';
import { ViewController } from '@/components/app/view-controller';
import { Toaster } from '@/components/ui/sonner';
import { useAgentErrors } from '@/hooks/useAgentErrors';
import { useDebugMode } from '@/hooks/useDebug';
import { getSandboxTokenSource } from '@/lib/utils';

const IN_DEVELOPMENT = process.env.NODE_ENV !== 'production';

const ALLOWED_HOSTS = [
  'youtube.com',
  'google.com',
  'wikipedia.org',
  'github.com',
  'bbc.co.uk',
  'reddit.com',
  'facebook.com',
  'twitter.com',
  'instagram.com',
];

function toAllowedUrl(raw: string): URL | null {
  try {
    const url = new URL(raw);
    if (url.protocol !== 'https:') return null;
    const ok = ALLOWED_HOSTS.some(
      (host) => url.hostname === host || url.hostname.endsWith('.' + host)
    );
    return ok ? url : null;
  } catch {
    return null;
  }
}

function AppSetup() {
  const room = useRoomContext();
  useDebugMode({ enabled: IN_DEVELOPMENT });
  useAgentErrors();

  useEffect(() => {
    room.localParticipant.registerRpcMethod('open_url', async (data) => {
      const url = toAllowedUrl(data.payload);
      if (!url) return 'blocked';

      const tab = window.open(url.href, '_blank');
      if (tab) {
        tab.opener = null;
      } else {
        toast(`Open ${url.hostname}?`, {
          action: {
            label: 'Open',
            onClick: () => window.open(url.href, '_blank'),
          },
        });
      }
      return 'ok';
    });
    return () => {
      room.localParticipant.unregisterRpcMethod('open_url');
    };
  }, [room]);

  return null;
}

interface AppProps {
  appConfig: AppConfig;
}

export function App({ appConfig }: AppProps) {
  const tokenSource = useMemo(() => {
    return typeof process.env.NEXT_PUBLIC_CONN_DETAILS_ENDPOINT === 'string'
      ? getSandboxTokenSource(appConfig)
      : TokenSource.endpoint('/api/token');
  }, [appConfig]);

  const session = useSession(
    tokenSource,
    appConfig.agentName ? { agentName: appConfig.agentName } : undefined
  );

  return (
    <AgentSessionProvider session={session}>
      <AppSetup />
      <JarvisBackground />
      <main className="relative z-10 grid h-svh grid-cols-1 place-content-center">
        <ViewController appConfig={appConfig} />
      </main>
      <StartAudioButton label="Start Audio" />
      <Toaster
        icons={{
          warning: <WarningIcon weight="bold" />,
        }}
        position="top-center"
        className="toaster group"
        style={
          {
            '--normal-bg': 'var(--popover)',
            '--normal-text': 'var(--popover-foreground)',
            '--normal-border': 'var(--border)',
          } as React.CSSProperties
        }
      />
    </AgentSessionProvider>
  );
}